import sys
sys.path.append('/local/home/weirchen/Research/projects/pips')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from pathlib import Path
import math
import cv2
import glob
import torch
import numpy as np
import torch.nn.functional as F
import imageio.v2 as imageio
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
import hydra
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt

# pips
from nets import Pips
from nets.raftnet import Raftnet
import saverloader
import utils.improc

# dpvo
from pips_slam import fastba, altcorr, lietorch
from pips_slam.lietorch import SE3
from pips_slam.stream import image_stream, video_stream, replica_stream
from pips_slam import projective_ops as pops
from pips_slam.plot_utils import plot_trajectory, save_trajectory_tum_format, save_pips_plot, save_edge_plot, load_gt_traj
from pips_slam.ba import BA

import pdb


def read_image(file):
    im = imageio.imread(file)
    im = im.astype(np.uint8)
    im = torch.from_numpy(im).permute(2,0,1)
    return im

def flatmeshgrid(*args, **kwargs):
    grid = torch.meshgrid(*args, **kwargs)
    return (x.reshape(-1) for x in grid)

def coords_grid_with_index(d, **kwargs):
    """ coordinate grid with frame index"""
    b, n, h, w = d.shape
    i = torch.ones_like(d)
    x = torch.arange(0, w, dtype=torch.float, **kwargs)
    y = torch.arange(0, h, dtype=torch.float, **kwargs)

    y, x = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    y = y.view(1, 1, h, w).repeat(b, n, 1, 1)
    x = x.view(1, 1, h, w).repeat(b, n, 1, 1)

    coords = torch.stack([x, y, d], dim=2)
    index = torch.arange(0, n, dtype=torch.float, **kwargs)
    index = index.view(1, n, 1, 1, 1).repeat(b, 1, 1, h, w)

    return coords, index

def ransac_mask(kpts0, kpts1, K0, K1, ransac, thresh=1.0, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    if ransac:
        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), threshold=norm_thresh,
            prob=conf,
            method=cv2.RANSAC)
    else:
        E, mask = cv2.findFundamentalMat(
            kpts0, kpts1,  method=cv2.FM_8POINT
        )
    
    # ret = None
    # if E is not None:
    #     best_num_inliers = 0

    #     for _E in np.split(E, len(E) / 3):
    #         n, R, t, _ = cv2.recoverPose(
    #             _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
    #         if n > best_num_inliers:
    #             best_num_inliers = n
    #             ret = (R, t[:, 0], mask.ravel() > 0)
    
    return mask.ravel() > 0

class PIPSSLAM:
    def __init__(self, cfg, ht=480, wd=640):
        self.cfg = cfg
        self.load_weights() 
        self.ht = ht
        self.wd = wd
        self.P = 1      # point: patch_size = 1
        self.S = cfg.model.S
        self.is_initialized = False
        self.enable_timing = False
        self.save_pips = cfg.save_pips
        self.pred_back = cfg.pred_back if 'pred_back' in cfg else None
        
        
        self.n = 0      # number of frames
        self.m = 0      # number of patches
        self.M = self.cfg.slam.PATCHES_PER_FRAME
        self.N = self.cfg.slam.BUFFER_SIZE
        
        # dummy image for visualization
        self.tlist = []
        self.counter = 0
        
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")  
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")
        
        self.patches_valid_ = torch.zeros(self.N, self.M, dtype=torch.float, device="cuda")
        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
    
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.targets = torch.zeros(1, 0, 2 , device="cuda")
        self.weights = torch.zeros(1, 0, 2 , device="cuda")
        
        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0
        
        self.local_window = []

        # for generating ground truth trajectory
        self.local_window_depth_g = []
        self.local_window_cam_g = []
        
        # store relative poses for removed frames
        self.delta = {}
    
        self.viewer = None
        if self.cfg.viz:
            self.start_viewer()

        # evaluation
        self.save_dir = os.path.join(self.cfg.output_dir, self.cfg.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics = {
            'ate': [],
            'ate_masked': []
        }
         
        
        # cache 
        self.cache_window = []
        # self.cache = {

        # }
        
        self.invalid_frames = []
        
    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, self.P, self.P)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)
        
    def load_weights(self):
        if self.cfg.model.mode == 'pips':
            self.network = Pips(S=self.cfg.model.S, stride=4).cuda()
            _ = saverloader.load(self.cfg.model.init_dir, self.network)
            self.network.eval()
        elif self.cfg.model.mode == 'raft':
            self.network = Raftnet(ckpt_name=self.cfg.model.init_dir).cuda()
            self.network.eval()
       
            

        
    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            intrinsics_)
        
    def preprocess(self, image, intrinsics):
        """ Load the image and store in the local window
        """      
        if len(self.local_window) >= self.S:
            self.local_window.pop(0)
        self.local_window.append(image)

        self.intrinsics_[self.n] = intrinsics
        
        # compute cache
        # with torch.no_grad():
        #     if self.cfg.model.mode == 'pips':
        #         if len(self.cache_window) >= self.S:
        #             self.cache_window.pop(0)
        #         image_ = 2 * (image / 255.0) - 1.0
        #         image_ = image_.unsqueeze(0)
        #         fmap = self.network.fnet(image_)
        #         self.cache_window.append(fmap)
        
        torch.cuda.empty_cache()
        
    def store_gt(self, depth_g, cam_g):
        """ Load the image and store in the local window
        """      
        if len(self.local_window_depth_g) >= self.S:
            self.local_window_depth_g.pop(0)
        self.local_window_depth_g.append(torch.from_numpy(depth_g))
        
        if len(self.local_window_cam_g) >= self.S:
            self.local_window_cam_g.pop(0)
        self.local_window_cam_g.append(torch.from_numpy(cam_g))
        
    def __image_gradient(self, images):
        # gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        gray = images.sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g
    
    def __image_gradient_2(self, images):
        images_pad = F.pad(images, (1,1,1,1), 'constant', 0)
        gray = images_pad.sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g
        
    def generate_patches(self, image):
        device = image.device
        B = 1
        # sample center
        # uniform
        if self.cfg.slam.PATCH_GEN == 'uniform':
            M_ = np.sqrt(self.M).round().astype(np.int32)
            grid_y, grid_x = utils.basic.meshgrid2d(B, M_, M_, stack=False, norm=False, device='cuda')
            grid_y = 8 + grid_y.reshape(B, -1)/float(M_-1) * (self.ht-16)
            grid_x = 8 + grid_x.reshape(B, -1)/float(M_-1) * (self.wd-16)
            coords = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
        elif self.cfg.slam.PATCH_GEN == 'random':
            x = torch.randint(1, self.wd-1, size=[1, self.M], device="cuda")
            y = torch.randint(1, self.ht-1, size=[1, self.M], device="cuda")
            coords = torch.stack([x, y], dim=-1).float()
        
        elif self.cfg.slam.PATCH_GEN == 'sift':
            margin = 16
            image_array = self.local_window[-1].permute(1, 2, 0).detach().cpu().numpy() # H, W, C
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            gray = ((gray - gray.min()) * (255.0 / (gray.max() - gray.min()))).astype(np.uint8)
            sift = cv2.SIFT_create()
            
            kps = sift.detect(gray,None)
            kps, des = sift.compute(gray, kps)                
            
            # Iterate over each keypoint
            kps_pt = [x.pt for x in kps]
            kp_np = np.array(kps_pt)
            if len(kps_pt) > 0:
                mask = (kp_np[...,0] > margin) & (kp_np[...,0] < self.wd - margin) & (kp_np[...,1] > margin) & (kp_np[...,1] < self.ht - margin) 
                kp_np = kp_np[mask]
                np.random.shuffle(kp_np)
                kp_np = kp_np[:self.M]

            if len(kp_np) == self.M:
                xys = torch.from_numpy(kp_np).to(device)
            else:
                # if detector is smaller than num_anchors, add random points
                diff = self.M - len(kp_np)
                x = torch.randint(margin, self.wd-margin, size=[diff])
                y = torch.randint(margin, self.wd-margin, size=[diff])
                xy = torch.stack([x, y], dim=-1).float().to(device)
                kp_np = torch.concat([xy, torch.from_numpy(kp_np).to(device)], dim=0)
                xys = kp_np

            coords = xys[None,...].float()    # self.M, 2

        # TODO: img_grad
        elif self.cfg.slam.PATCH_GEN == 'img_grad':
            margin = 64

            g = self.__image_gradient(self.local_window[-1][None, None, ...])
            x = torch.randint(margin, self.wd-margin, size=[1, 3*self.M], device="cuda")
            y = torch.randint(margin, self.ht-margin, size=[1, 3*self.M], device="cuda")

            coords = torch.stack([x, y], dim=-1).float()    ## [1, N, 2]
            g = altcorr.patchify(g[0,:,None], coords, 0).view(1, 3 * self.M)
            
            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -self.M:])
            y = torch.gather(y, 1, ix[:, -self.M:])
            coords = torch.stack([x, y], dim=-1).float()
            # pdb.set_trace()    
            
        elif 'grid_grad' in self.cfg.slam.PATCH_GEN:
            rel_margin = 0.15
            num_expand = 8
            
            grid_size = int(self.cfg.slam.PATCH_GEN.split('_')[-1])
            num_grid = grid_size * grid_size
            grid_M = self.M // num_grid
            H_grid, W_grid = self.ht//grid_size, self.wd // grid_size
            
            g = self.__image_gradient_2(self.local_window[-1][None, None, ...])
      
            x = torch.rand((num_grid, num_expand * grid_M), device="cuda") * (1 - 2 * rel_margin) + rel_margin
            y = torch.rand((num_grid, num_expand * grid_M), device="cuda") * (1 - 2 * rel_margin) + rel_margin
            # map to coordinate
            offset = torch.linspace(0, grid_size-1, grid_size)
            offset_y, offset_x = torch.meshgrid(offset, offset)
            offset = torch.stack([offset_x, offset_y], dim=-1).to('cuda')
            offset = offset.view(-1,2)
            offset[...,0] = offset[...,0] * W_grid
            offset[...,1] = offset[...,1] * H_grid
        
            x_global = x.view(1, num_grid, -1) *W_grid  +  offset[...,0].view(1,-1,1) 
            y_global = y.view(1, num_grid, -1) *H_grid  +  offset[...,1].view(1,-1,1) 
        
            coords = torch.stack([x_global, y_global], dim=-1).float()    ## [1, N, 2]
            coords = rearrange(coords, 'b g n c -> b (g n) c')
            coords = torch.round(coords).unsqueeze(1)
            coords_norm = coords
            coords_norm[...,0] = coords_norm[...,0] / (self.wd - 1) * 2.0 - 1.0
            coords_norm[...,1] = coords_norm[...,0] / (self.ht - 1) * 2.0 - 1.0

            gg = F.grid_sample(g, coords_norm, mode='bilinear', align_corners=True)
            gg = gg[:,0,0]
            gg = rearrange(gg,'b (ng n) -> b ng n', ng=num_grid)
            ix = torch.argsort(gg, dim=-1)         
            x_global = torch.gather(x_global, 2, ix[:, :, -grid_M:])
            y_global = torch.gather(y_global, 2, ix[:, :, -grid_M:])
            coords = torch.concat([x_global, y_global], dim=-1).float()

        disps = torch.ones(B, 1, self.ht, self.wd, device="cuda")
        grid, _ = coords_grid_with_index(disps, device=self.poses_.device) 
        patches = altcorr.patchify(grid[0], coords, self.P//2).view(B, -1, 3, self.P, self.P)  # B, N, 3, p, p

        clr = altcorr.patchify(image.unsqueeze(0).float(), (coords + 0.5), 0).view(B, -1, 3)

        return patches, clr
    
    def init_motion(self):
        if self.n > 1:
            if self.cfg.slam.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n-1])
                P2 = SE3(self.poses_[self.n-2])
                
                xi = self.cfg.slam.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.poses_[self.n] = tvec_qvec
    
    def append_factors(self, ii, jj):
        """Add edges to factor graph 

        Args:
            ii (_type_): patch idx
            jj (_type_): frame idx
        """
        # project patch k from i to j
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        # self.ix = self.index_
        self.ii = torch.cat([self.ii, self.ix[ii]])
    

        # print("append_factors")
        # print("jj", jj, jj.shape)
        # print("kk", ii, ii.shape)
        # print("ii", self.ix[ii], self.ix[ii].shape)
        # print("self.n", self.n, "self.ii", self.ii.shape)

    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.targets = self.targets[:,~m]
        self.weights = self.weights[:,~m]
        
    def __edges_forw(self):
        """From frame0's patches to all local frames (including frame 0)

        Returns:
            _type_: _description_
        """
        t0 = self.M * max((self.n - self.S), 0)
        t1 = self.M * max((self.n - self.S + 1), 1)

        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-self.S, 0), self.n, device="cuda"), indexing='ij')
        
    def __edges_back(self):
        """Edge between current patches and the previous edge
        """
        r=self.cfg.slam.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        # print("__edges_back", t0, t1, max(self.n-r, 0), self.n)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-self.S, 0), self.n, device="cuda"), indexing='ij')


    def delete_edges(self):
        pass
    
    
    def get_gt_trajs(self, xy):
        """Compute the gt trajectories from ground truth depth and camera pose

        Args:
            xy (tensor): B, N, 2
        """
        B, N = xy.shape[:2]
        
        depths = torch.stack(self.local_window_depth_g, dim=0).unsqueeze(0).to(xy.device)   # B, S, C, H, W
        cams_c2w = torch.stack(self.local_window_cam_g, dim=0).unsqueeze(0).to(xy.device)   # B, S, C, H, W
        intrinsics = self.intrinsics[:,max(0,self.n-self.S):self.n].to(xy.device)
        
        assert len(self.local_window_cam_g) == len(self.local_window_depth_g)
        if self.pred_back:
            depths = torch.flip(depths, dims=(1,))
            cams_c2w = torch.flip(cams_c2w, dims=(1,))
            intrinsics = torch.flip(intrinsics, dims=(1,))
            
        if depths.shape[1] < self.S:
            repeat_depths = repeat(depths[:,-1], 'b h w -> b s h w', s=self.S-depths.shape[1])
            depths = torch.cat([depths, repeat_depths], dim=1)
            
        if cams_c2w.shape[1] < self.S:
            repeat_cams_c2w = repeat(cams_c2w[:,-1], 'b h w -> b s h w', s=self.S-cams_c2w.shape[1])
            cams_c2w = torch.cat([cams_c2w, repeat_cams_c2w], dim=1)
        
        if intrinsics.shape[1] < self.S:
            repeat_intrinsics = repeat(intrinsics[:,-1], 'b c -> b s c', s=self.S-intrinsics.shape[1])
            intrinsics = torch.cat([intrinsics, repeat_intrinsics], dim=1)
            
        # back-project xy from frame 0
        xy_depth = altcorr.patchify(depths[:,[0]].float(), xy, 0).reshape(B, N, 1)
        P0 = pops.back_proj(xy, xy_depth, intrinsics[:,0], cams_c2w[:,0])

        # project to all frame in the local window
        # pdb.set_trace()
        cams_w2c = torch.inverse(cams_c2w)
        xy_gt = pops.proj_to_frames(P0, intrinsics, cams_w2c)
        
        xy_gt = xy_gt[:,:len(self.local_window_depth_g)]
        if self.pred_back:
            xy_gt = torch.flip(xy_gt, dims=(1,))
            
        # Detect NAN value
        xy_repeat = repeat(xy, 'b n c -> b s n c', s=len(self.local_window_depth_g))
        invalid = torch.isnan(xy_gt) | torch.isinf(xy_gt)
        invalid_depth = (xy_depth <= 0) | torch.isnan(xy_depth) | torch.isinf(xy_depth)
        invalid_depth = repeat(invalid_depth, 'b n i -> b s n (i c)', s=len(self.local_window_depth_g), c=2)
        invalid = invalid | invalid_depth
        xy_gt[invalid] = xy_repeat[invalid]
        valid = ~invalid
        
        # # DEBUG
        # if not (invalid.any(dim=1).any(dim=-1) == (xy_depth == 0)).all():
        #     pdb.set_trace()
        
        return xy_gt, valid
                
    def get_window_trajs(self, only_coords=False):
        rgbs = torch.stack(self.local_window, dim=0).unsqueeze(0)   # B, S, C, H, W
        # if self.cfg.model.mode == 'pips':
        #     fmaps = torch.stack(self.cache_window, dim=1)

        if self.pred_back:
            rgbs = torch.flip(rgbs, dims=(1,))
            # if self.cfg.model.mode == 'pips':
            #     fmaps = torch.flip(fmaps, dims=(1,))
            xy = self.patches_[self.n-1, :, :2, self.P//2, self.P//2]
            xy = xy.unsqueeze(0) # B, M, 2
        else:
            # get coords of the first frame in local window
            xy = self.patches_[self.n-1, :, :2, self.P//2, self.P//2]
            xy = xy.unsqueeze(0) # B, M, 2
        
        if only_coords:
            return xy
        
        # pad repeated frames to make local window = S
        if rgbs.shape[1] < self.S:
            repeat_rgbs = repeat(rgbs[:,-1], 'b c h w -> b s c h w', s=self.S-rgbs.shape[1])
            rgbs = torch.cat([rgbs, repeat_rgbs], dim=1)

            # if self.cfg.model.mode == 'pips':
            #     fmaps = torch.stack(self.cache_window, dim=1)
            #     repeat_fmaps = repeat(fmaps[:,-1], 'b c h w -> b s c h w', s=self.S-fmaps.shape[1])
            #     fmaps = torch.cat([fmaps, repeat_fmaps], dim=1)

        if self.cfg.model.mode == 'pips':
            preds, preds_anim, vis_e, stats = self.network(xy, rgbs, iters=self.cfg.model.I)
            # preds, preds_anim, vis_e, stats = self.network(xy, rgbs, fmaps=fmaps, iters=self.cfg.model.I)
            local_target = preds[-1]
            vis_label = (vis_e > 0.5)   # B, S, N

        elif self.cfg.model.mode == 'raft':
            prep_rgbs = utils.improc.preprocess_color(rgbs)
            flows_e = []
            for s in range(self.S-1):
                rgb0 = prep_rgbs[:,s]
                rgb1 = prep_rgbs[:,s+1]
                flow, _ = self.network(rgb0, rgb1, iters=32)
                flows_e.append(flow)
            flows_e = torch.stack(flows_e, dim=1) # B, S-1, 2, H, W
            coords = []
            coord0 = xy # B, N_*N_, 2
            coords.append(coord0)
            coord = coord0.clone()
            for s in range(self.S-1):
                delta = utils.samp.bilinear_sample2d(
                    flows_e[:,s], coord[:,:,0], coord[:,:,1]).permute(0,2,1) # B, N, 2, forward flow at the discrete points
                coord = coord + delta
                coords.append(coord)
            local_target = torch.stack(coords, dim=1) # B, S, N, 2
            
            vis_label = torch.ones(local_target.shape[:3]).bool()
            # Path("saved_graphs").mkdir(exist_ok=True)
            # save_edge_plot(self.ii.detach().cpu().numpy(), self.jj.detach().cpu().numpy(), self.kk.detach().cpu().numpy(), self.n, 'saved_graphs')
        
        local_target = local_target[:,:len(self.local_window)]
        vis_label = vis_label[:,:len(self.local_window)]
        
        if self.pred_back:
            local_target = torch.flip(local_target, dims=(1,))
            vis_label = torch.flip(vis_label, dims=(1,))
            
        return local_target, vis_label, xy


    def eval_ate(self, trajs, trajs_gt, pred_mask, gt_mask):
        ate = torch.norm(trajs - trajs_gt, dim=-1) # B, S, N
        B, S, N = ate.shape
        # padding = 0
        # mask = (trajs_gt[...,0] >= padding) & (trajs_gt[...,0] < self.wd - padding) & (trajs_gt[...,1] >= padding) & (trajs_gt[...,1] < self.ht - padding) 
        mask = pred_mask & gt_mask
        ate_masked = ate[mask].mean().detach().cpu().numpy()
        self.metrics['ate_masked'].append(ate_masked)
        self.metrics['ate'].append(ate.mean().detach().cpu().numpy())
        return ate_masked
    
    def compute_ransac_mask(self, trajs):
        """Compute the ransac mask based on trajectories
        
        Args:
            trajs: B, S, N, C
        Returns:    
            masks: B, S, N
        """
        
        B, S, N, C = trajs.shape
        assert B==1
        trajs_np = trajs.detach().cpu().numpy()
        intrinsics = self.intrinsics[:,max(0,self.n-self.S):self.n].detach().cpu().numpy()
        Ks = np.zeros((B, S, 3, 3))
        Ks[:,:,0,0] = intrinsics[...,0]
        Ks[:,:,1,1] = intrinsics[...,1]
        Ks[:,:,0,2] = intrinsics[...,2]
        Ks[:,:,1,2] = intrinsics[...,3]
        Ks[:,:,2,2] = 1
        
        if self.pred_back:
            ref_idx = S - 1
        else:
            ref_idx = 0
        masks = np.ones((B, S, N))
        for i in range(S):
            pts1 = trajs_np[0,i]
            pts2 = trajs_np[0, ref_idx]
            mask = ransac_mask(pts1, pts2, Ks[0,i], Ks[0,ref_idx], ransac=True)
            masks[0,i] = mask.reshape(-1)
  
        masks = torch.from_numpy(masks).to(trajs.device).bool()
        return masks
        
    def predict_target(self):
        # predict target
        with torch.no_grad():
            if self.cfg.use_gt_traj:
                coords_xy = self.get_window_trajs(only_coords=True)
            else:
                trajs, vis_label, coords_xy = self.get_window_trajs()
            
        trajs_gt, valid_gt = self.get_gt_trajs(coords_xy)
        
        if self.cfg.use_gt_traj:
            trajs = trajs_gt
            vis_label = torch.ones(trajs_gt.shape[:3]).bool()
        
        # rearrange s.t. it matches the edge order
        B, S, N, C = trajs.shape
        local_target = rearrange(trajs, 'b s n c -> b (n s) c')
        
        # predict weight 
        if valid_gt.float().mean() < 1.0:
            self.invalid_frames.append(self.n)

        if self.cfg.use_gt_traj:
            local_weight = rearrange(valid_gt.float(), 'b s n c -> b (n s) c')
        else:
            local_weight = torch.ones_like(local_target)
        
        # apply traj_pred
        vis_label = rearrange(vis_label, 'b s n -> b (n s)')
        local_weight[~vis_label] = 0
        
        # edge_weight_decay: the closer to the reference frame, the larger the weight
        if 'EDGE_WEIGHT_DECAY' in self.cfg.slam:
            # local_weight = rearrange(local_weight, 'b (n s) c -> b s n c', s=S, n=N)
            edge_weight_decay = torch.ones(S).to(local_weight.device)
            edge_weight_decay[1:] = self.cfg.slam.EDGE_WEIGHT_DECAY
            edge_weight_decay_cum = torch.cumprod(edge_weight_decay, dim=0)
            if self.pred_back:
                edge_weight_decay_cum = torch.flip(edge_weight_decay_cum, dims=(0,))
            edge_weights = repeat(edge_weight_decay_cum, 's -> b (n s) c', b=B, c=C, n=N)
            local_weight = local_weight * edge_weights

        # compute ransac masks
        if self.cfg.slam.USE_RANSAC:
            ransac_mask = self.compute_ransac_mask(trajs)
            ransac_mask = repeat(ransac_mask, 'b s n -> b (n s) c', c=C)
            local_weight[~ransac_mask] = 0
            
        # for target output boundary, set weight to 0
        padding = 20
        boundary_mask = (local_target[...,0] >= padding) & (local_target[...,0] < self.wd - padding) & (local_target[...,1] >= padding) & (local_target[...,1] < self.ht - padding) 
        local_weight[~boundary_mask] = 0 
        
        # GT FILTERING: using GT traj to filter pred
        if 'GT_FILTERING' in self.cfg.slam and self.cfg.slam.GT_FILTERING > 0:
            ate = torch.norm(trajs - trajs_gt, dim=-1) # B, S, N
            ate_mask = rearrange((ate < self.cfg.slam.GT_FILTERING), 'b s n -> b (n s)')
            local_weight[~ate_mask] = 0    
            gt_mask = rearrange(valid_gt, 'b s n c -> b (n s) c')        
            local_weight[~gt_mask] = 0
        
        # check if some frame has too fewer matches
        if 'MIN_VALID_PATCH' in self.cfg.slam and self.cfg.slam.MIN_VALID_PATCH > 0:
            patch_valid = (local_weight > 0).any(dim=-1)    
            patch_valid = rearrange(patch_valid, 'b (n s) -> b s n', s=S, n=N)
            frame_valid = (patch_valid.sum(dim=2) >= self.cfg.slam.MIN_VALID_PATCH)
            frame_mask = repeat(frame_valid, 'b s -> b (n s)', n=N)
            local_weight[~frame_mask] = 0
            
        # check if the patches as some optimization targets
        if self.n >= self.cfg.slam.MIN_TRACK_LEN:
            patch_valid = (local_weight > 0).any(dim=-1)
            patch_valid = rearrange(patch_valid, 'b (n s) -> b s n', s=S, n=N)
            patch_valid = (patch_valid.sum(dim=1) >= self.cfg.slam.MIN_TRACK_LEN)
            self.patches_valid_[self.n-1] = patch_valid.squeeze(0)
            track_len_mask = repeat(patch_valid, 'b n -> b (n s)', s=S)
            local_weight[~track_len_mask] = 0
                
                
        # append to global targets, weights
        self.targets = torch.cat([self.targets, local_target], dim=1)
        self.weights = torch.cat([self.weights, local_weight], dim=1)

        # evaluate 
        valid_gt = (valid_gt).any(dim=-1)
        valid_pred = rearrange((local_weight > 0).any(dim=-1), ' b (n s) -> b s n', n=N, s=S)
        
        if self.save_pips and self.n > 1:
            rgbs = torch.stack(self.local_window, dim=0).unsqueeze(0) 
            if torch.isnan(trajs_gt).any():
                pdb.set_trace()
            save_pips_path = os.path.join(self.save_dir,'saved_images')
            Path(save_pips_path).mkdir(exist_ok=True)
            save_pips_plot(
                rgbs[0].detach().cpu().numpy(), 
                trajs[0].detach().cpu().numpy(),
                trajs_gt[0].detach().cpu().numpy(), 
                self.n, 
                save_pips_path, 
                start_idx=max(0, self.n-self.S),
                valid_label=valid_pred[0].detach().cpu().numpy(),
                valid_gt=valid_gt[0].detach().cpu().numpy()
            )
            
        # FIXME:
        # save_trajs_dir = os.path.join(self.save_dir,'saved_trajs')
        # Path(save_trajs_dir).mkdir(exist_ok=True)
        # save_trajs_path = os.path.join(save_trajs_dir, f"{self.n}")
        # np.save(save_trajs_path, trajs.detach().cpu().numpy())

        self.eval_ate(trajs, trajs_gt, valid_pred, valid_gt)
    
    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()
    
    
    def map_point_filtering(self):
        coords = self.reproject()[...,self.P//2, self.P//2]
        ate = torch.norm(coords - self.targets,dim=-1)
        reproj_mask = (ate < self.cfg.slam.MAP_FILTERING_TH)
        self.weights[~reproj_mask] = 0
    
    def normalize_scale(self):
        # For the valid patch_points, set the median scale to be 1
        valid_patches = self.patches_[:self.n][(self.patches_valid_ > 0)[:self.n]]  
        depths = valid_patches[:,2]
        median_depth = torch.median(depths)
        
        scale = 1.0 / median_depth
        
        # normalized patches depth
        self.patches_[:self.n,:,2] *= scale
        self.poses_[:self.n, :3] *= scale
        
        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
        points = (points[...,self.P//2,self.P//2,:3] / points[...,self.P//2,self.P//2,3:]).reshape(-1, 3)
        self.points_[:len(points)] = points[:]
        
        
    def update(self):
        # lmbda
        lmbda = torch.as_tensor([1e-4], device="cuda")
        
        # ba
        t0 = self.n - self.cfg.slam.OPTIMIZATION_WINDOW if self.is_initialized else 1
        t0 = max(t0, 1)
        print(f"t0: {t0}, t1: {self.n}, weights: {self.weights.mean()}")
        

        # RES = 4.0
        # self.poses_ /= RES
        # self.patches_ /= RES
        # self.intrinsics_ /= RES 
        # self.targets /= RES

        # fastba.BA(self.poses, self.patches, self.intrinsics, 
            # self.targets.detach(), self.weights.detach(), lmbda, self.ii, self.jj, self.kk, t0, self.n, self.cfg.slam.ITER)

        # self.poses_ *= RES
        # self.patches_ *= RES
        # self.intrinsics_ *= RES 
        # self.targets *= RES

        # change to normal BA

        # if self.n % 50 == 0:
        #     pdb.set_trace()

        ep = 10
        lmbda = 1e-4
        bounds = [0, 0, self.wd, self.ht]
        Gs = SE3(self.poses)
        patches = self.patches
        for itr in range(self.cfg.slam.ITER):
            Gs, patches = BA(Gs, patches, self.intrinsics.detach(), self.targets.detach(), self.weights.detach(), lmbda, self.ii, self.jj, self.kk, 
                bounds, ep=ep, fixedp=t0, structure_only=False, loss=self.cfg.slam.LOSS)
        
        # for keeping the same memory -> viewer works
        self.patches_[:] = patches.reshape(self.N, self.M, 3, self.P, self.P)
        self.poses_[:] = Gs.vec().reshape(self.N, 7)
        
        # 3D points culling
        if self.cfg.slam.USE_MAP_FILTERING:
            with torch.no_grad():
                self.map_point_filtering()
        
        # TODO: debug extracting point
        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
        points = (points[...,self.P//2,self.P//2,:3] / points[...,self.P//2,self.P//2,3:]).reshape(-1, 3)
        self.points_[:len(points)] = points[:]

        
    def terminate(self):
        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float)

        if self.viewer is not None:
            self.viewer.join()

        # save metrics
        save_metrics_path = os.path.join(self.save_dir,'saved_metrics')
        Path(save_metrics_path).mkdir(exist_ok=True)
        for name, metric_list in self.metrics.items():
            np.save(f'{save_metrics_path}/{name}', np.array(metric_list))

        return poses, tstamps
    
    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def keyframe(self):
        if self.pred_back:
            to_remove = self.ix[self.kk] < self.n - self.cfg.slam.REMOVAL_WINDOW
        else:
            to_remove = self.ix[self.kk] < (self.n - self.cfg.slam.REMOVAL_WINDOW - self.S)
        
        self.remove_factors(to_remove)
    
    def eval_pose():
        pass
    
    def __call__(self, tstamp, image, intrinsics, depth_g=None, cam_g=None):
        """main function of tracking

        Args:
            tstamp (_type_): _description_
            image (_type_): 3, H, W
            intrinsics (_type_): fx, fy, cx, cy

        Raises:
            Exception: _description_
        """
        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image)

        # image preprocessing   
        self.preprocess(image, intrinsics)
        
        # for debug
        self.store_gt(depth_g, cam_g)
        
        # print("local_window", len(self.local_window))
        
        # generate patches
        patches, clr = self.generate_patches(image)
        
        # depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s
        
        self.patches_[self.n] = patches      
        self.patches_valid_[self.n] = 1
        # color info for visualization
        # clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        # self.colors_[self.n] = clr.to(torch.uint8)
                
        # pose initialization with motion model
        self.init_motion()
                
        # update states
        # self.index_[self.n + 1] = self.n + 1
        # self.index_map_[self.n + 1] = self.m + self.M
        # initialize the mapping of the current frame
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter

        
        # color info for visualization
        # clr = clr[0,:,[2,1,0]]
        clr = clr[0]
        self.colors_[self.n] = clr.to(torch.uint8)

        self.index_[self.n] = self.n
        if self.pred_back:
            self.index_[self.n] = self.n
        else:
            self.index_[self.n] = max(self.n - self.S, 0)
            
        self.index_map_[self.n] = self.m
        
        self.counter += 1    
        
        self.n += 1
        self.m += self.M
        
        # ba
        # if self.n == 8 and not self.is_initialized:
        #     self.is_initialized = True
        #     if self.pred_back:
        #         self.append_factors(*self.__edges_back())
        #     else:
        #         self.append_factors(*self.__edges_forw())
        #     for itr in range(1):
        #         self.update()
        
        # elif self.is_initialized:
        #     if self.pred_back:
        #         self.append_factors(*self.__edges_back())
        #     else:
        #         self.append_factors(*self.__edges_forw())
        #     self.update()
        #     self.keyframe()

        if self.pred_back:
            self.append_factors(*self.__edges_back())
        else:
            self.append_factors(*self.__edges_forw())
        self.predict_target()
        
        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True            
            # one initialized, run global BA
            for itr in range(12):
                self.update()
            
            if self.cfg.slam.USE_SCALE_NORM:
                self.normalize_scale()

        elif self.is_initialized:
            self.update()
            self.keyframe()


        # self.eval_pose()
        self.print_stats()

        if self.cfg.save_results:
            self.save_results()

        torch.cuda.empty_cache()

    def print_stats(self):
        line = "[metrics]"
        for name, metric_list in self.metrics.items():
            # line += f" {name}: {np.mean(metric_list)}"
            line += f" {name}: {metric_list[-1]}"
        print(line)

        # print("invalid", self.invalid_frames)
    
    def save_results(self):
        """Save the camera pose and 3D point at the moment
        """
        Gs = SE3(self.poses_).detach()
        pose = Gs.matrix()[:self.n].float().detach().cpu().numpy()
        pts = self.points_[:self.n * self.M].float().detach().cpu().numpy()
        clr = self.colors_[:self.n].reshape(-1,3).float().detach().cpu().numpy()
        
        pts_valid = self.patches_valid_[:self.n].reshape(-1).detach().cpu().numpy()

        save_results_path = os.path.join(self.save_dir,'saved_results')
        Path(save_results_path).mkdir(exist_ok=True)
        save_name = os.path.join(save_results_path, f'{self.n}.npz')
        np.savez(save_name, pose=pose, pts=pts, clr=clr, pts_valid=pts_valid)
    



def get_replica_intrinsics():
    # intrinsics
    img_w, img_h = 640, 480
    hfov = 90
    # the pin-hole camera has the same value for fx and fy
    fx = img_w / 2.0 / math.tan(math.radians(hfov / 2.0))
    # self.fy = self.H / 2.0 / math.tan(math.radians(self.yhov / 2.0))
    fy = fx
    cx = (img_w - 1.0) / 2.0
    cy = (img_h - 1.0) / 2.0
    return fx, fy, cx, cy 


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    B = 1
    S = 8
    N = 16 ** 2
    
    # dataset_name = 'replica'
    # dataset_root = '/media/weirchen/T7/datasets/Replica_Dataset'
    # scene = 'office_0/Sequence_1'
    
    # max_len = 10
    # stride = 1
    
    # modeltype = 'pips'
    # init_dir = '/local/home/weirchen/Research/projects/pips/reference_model'
    # filenames = glob.glob(f'{dataset_root}/{scene}/rgb/*.png')
    # if max_len > 0:
    #     filenames = filenames[:max_len]
    # filenames = filenames[::stride]
    # print('filenames', filenames[:S], len(filenames), "stride", stride, "max_len", max_len)
    
    slam = None
    timeit = False
    skip = 0
    
    # queue = Queue(maxsize=1)


    imagedir, calib, stride, skip = cfg.data.imagedir, cfg.data.calib, cfg.data.stride, cfg.data.skip 
    # if os.path.isdir(cfg.data.imagedir):
    #     reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    # else:
    #     reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    # reader.start()
    # if os.path.isdir(cfg.data.imagedir):
    #     dataloader = image_stream(None, imagedir, calib, stride, skip)
    # else:
    #     dataloader = video_stream(None, imagedir, calib, stride, skip)
    dataloader = replica_stream(imagedir, calib, stride, skip)
    
    for i, (t, image, depth, intrinsics, cam_c2w) in enumerate(dataloader):

        if "max_length" in cfg.data and i >= cfg.data.max_length: break
        if t < 0: break
        
        print(i, image.shape, depth.shape)
        
        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()
        
        
        # initialization
        if slam is None:
            slam = PIPSSLAM(cfg, ht=image.shape[1], wd=image.shape[2])
        # pdb.set_trace()
        # tracking
        slam(t, image, intrinsics, depth_g=depth, cam_g=cam_c2w)

    # tuple (N, 7), (N, s)
    pred_traj = slam.terminate()
    
    gt_traj_file = os.path.join(cfg.data.imagedir, 'traj_w_c.txt')
    gt_traj = load_gt_traj(gt_traj_file, skip=cfg.data.skip, stride=cfg.data.stride)
    
    if cfg.save_trajectory:
        # Path(os.path.join(slam.save_dir, "saved_trajectories")).mkdir(exist_ok=True)
        # save_trajectory_tum_format(pred_traj, f"saved_trajectories/{cfg.exp_name}.txt")
        save_trajectory_tum_format(pred_traj, os.path.join(slam.save_dir,'traj.txt'))

    if cfg.plot:
        # Path("trajectory_plots").mkdir(exist_ok=True)
        plot_trajectory(pred_traj, gt_traj=gt_traj, title=f"PIP_SLAM Trajectory Prediction for {cfg.exp_name}", filename=os.path.join(slam.save_dir,'traj_plot.pdf'))
        
if __name__ == '__main__':
    main()