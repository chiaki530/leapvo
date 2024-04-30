# import sys
# sys.path.append('/local/home/weirchen/Research/projects/pips')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import matplotlib
matplotlib.use('Agg')

from pathlib import Path
import math
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from omegaconf import DictConfig
import hydra


from leapvo.leap.cotracker_kernel_v2 import CoTrackerKernelV2


# dpvo
from leapvo.backend import altcorr, lietorch
from leapvo.backend.lietorch import SE3
from leapvo.stream import sintel_stream, dataset_stream
from leapvo.backend import projective_ops as pops
from leapvo.backend.ba import BA

from leapvo.plot_utils import plot_trajectory, save_trajectory_tum_format, save_pips_plot, eval_metrics, load_gt_traj, load_traj, load_timestamps

from leapvo.slam_visualizer import CoTrackerSLAMVisualizer
from leapvo.timer import Timer
from leapvo.rerun_visualizer import vis_rerun
import pdb
import time

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
    return mask.ravel() > 0
    

class COTRACKERSLAM:
    def __init__(self, cfg, ht=480, wd=640):
        # super(COTRACKERSLAM, self).__init__(cfg=cfg, ht=ht, wd=wd)

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

        
        self.S_model = cfg.model.S
        self.S_slam = cfg.slam.S_slam       # tracked window
        self.S = cfg.slam.S_slam       
        self.kf_stride = cfg.slam.kf_stride
        self.interp_shape = (384, 512)


        self.load_gt = cfg.load_gt
        save_dir = f"{cfg.data.savedir}/{cfg.data.name}"

        self.use_forward = cfg.slam.use_forward if 'use_forward' in cfg.slam else True
        self.use_backward = cfg.slam.use_backward if 'use_backward' in cfg.slam else True
        
        print("[use_forward]", self.use_forward,'[use_backward]', self.use_backward)
        self.visualizer = CoTrackerSLAMVisualizer(cfg, save_dir=save_dir)
    
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

    def __image_gradient_2(self, images):
        images_pad = F.pad(images, (1,1,1,1), 'constant', 0)
        gray = images_pad.sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g
    
    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)
    
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
    
    def map_point_filtering(self):
        coords = self.reproject()[...,self.P//2, self.P//2]
        ate = torch.norm(coords - self.targets,dim=-1)
        reproj_mask = (ate < self.cfg.slam.MAP_FILTERING_TH)
        self.weights[~reproj_mask] = 0
    
    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()
    
    def load(self):
        strict = True
        if self.cfg.model.init_dir != "":
            state_dict = torch.load(self.cfg.model.init_dir, map_location='cuda:0')
            if "model" in state_dict:
                state_dict = state_dict["model"]

            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            self.network.load_state_dict(state_dict, strict=strict)


    def load_weights(self):
        if self.cfg.model.mode == 'cotracker_kernel_v2':
            self.network = CoTrackerKernelV2(cfg=self.cfg, stride=self.cfg.model.stride).cuda()
            self.load()
            self.network.eval()
        else:
            raise NotImplementedError

    def preprocess(self, image, intrinsics):
        """ Load the image and store in the local window
        """      
        if len(self.local_window) >= self.S:
            self.local_window.pop(0)
        self.local_window.append(image)

        self.intrinsics_[self.n] = intrinsics
        
        torch.cuda.empty_cache()
    

    def __edges(self):
        """Edge between keyframe patches and the all local frames
        """
        r = self.cfg.slam.S_slam
        local_start_fid = max((self.n - r), 0)
        local_end_fid = max((self.n - 0), 0)
        idx = torch.arange(0, self.n * self.M, device="cuda").reshape(self.n, self.M)
        kf_idx = idx[local_start_fid:local_end_fid:self.kf_stride].reshape(-1)
        
        # print('local_window', local_start_fid, local_end_fid)
        # print("kf_idx", kf_idx, kf_idx.shape)
        return flatmeshgrid(
            kf_idx,
            torch.arange(max(self.n-self.S_slam, 0), self.n, device="cuda"), indexing='ij')
    
    def __edges_forw(self):
        """Edge between previous patches and the current frame
        """
        r=self.cfg.slam.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        """Edge between current patches and the previous edge
        """
        r=self.cfg.slam.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        # print("__edges_back", t0, t1, max(self.n-r, 0), self.n)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-self.S, 0), self.n, device="cuda"), indexing='ij')


    def get_gt_trajs(self, xys, xys_sid):
        """Compute the gt trajectories from ground truth depth and camera pose

        Args:
            xys (tensor): B, N, 2
            xys_sid (tensor): B, N
        Returns:
            xy_gt (tensor): B, S, N, 2
            valid (tensor): B, S, N, 2
        """
        B, N = xys.shape[:2]
        S = len(self.local_window_depth_g)
   
        depths = torch.stack(self.local_window_depth_g, dim=0).unsqueeze(0).to(xys.device)   # B, S, C, H, W
        cams_c2w = torch.stack(self.local_window_cam_g, dim=0).unsqueeze(0).to(xys.device)   # B, S, C, H, W
        intrinsics = self.intrinsics[:,self.n-S:self.n].to(xys.device)
        
        assert len(self.local_window_cam_g) == len(self.local_window_depth_g)

        # back-project xy from each frame
        P0 = torch.empty(B, N, 4).to(xys.device)
        xy_depth = torch.empty(B, N, 1).to(xys.device)
        for s in range(S):
            mask = (xys_sid == s)
            xys_s = xys[mask].reshape(B, self.M, 2)
            depth_s = altcorr.patchify(depths[:,[s]].float(), xys_s, 0).reshape(B, self.M, 1)
            xy_depth[mask] = depth_s.reshape(-1, 1)
            P0[mask] = pops.back_proj(xys_s, depth_s, intrinsics[:,s], cams_c2w[:,s]).reshape(-1, 4)

        # project to all frame in the local window
        cams_w2c = torch.inverse(cams_c2w)
        xy_gt = pops.proj_to_frames(P0, intrinsics, cams_w2c)
        
        xy_gt = xy_gt[:,:S]
            
        # Detect NAN value
        xy_repeat = repeat(xys, 'b n c -> b s n c', s=S)
        invalid = torch.isnan(xy_gt) | torch.isinf(xy_gt)
        invalid_depth = (xy_depth <= 0) | torch.isnan(xy_depth) | torch.isinf(xy_depth)
        invalid_depth = repeat(invalid_depth, 'b n i -> b s n (i c)', s=S, c=2)
        invalid = invalid | invalid_depth
        xy_gt[invalid] = xy_repeat[invalid]
        valid = ~invalid

        return xy_gt, valid
    
    def get_queries(self):
        """return the query of the current local video window

        Returns:
            queries: (1, N, 3) in format (t, x, y)
        """

        S = len(self.local_window)
        xys = self.patches_[self.n-S:self.n, :, :2, self.P//2, self.P//2] 
        xys = xys.unsqueeze(0) # B, S, M, 2
        
        B = xys.shape[0]
         # compute xys_sid
        xys_sid = repeat(torch.arange(S).to(xys.device), 's -> b s m', b=B, m=self.M)
        
        xys = rearrange(xys[:, ::self.kf_stride], 'b s m c -> b (s m) c')
        xys_sid = rearrange(xys_sid[:,::self.kf_stride], 'b s m -> b (s m)')

        queries = torch.cat([xys_sid.unsqueeze(-1), xys], dim=2)

        return queries


    def get_patches_xy(self):
        S = len(self.local_window)
        # extract the patches from local windows 
        xys = self.patches_[self.n-S:self.n, :, :2, self.P//2, self.P//2]  # S, M, 2
        xys = xys.unsqueeze(0) # B, S, M, 2
        
        B = xys.shape[0]
         # compute xys_sid
        xys_sid = repeat(torch.arange(S).to(xys.device), 's -> b s m', b=B, m=self.M)
        xys = rearrange(xys, 'b s m c -> b (s m) c')
        xys_sid = rearrange(xys_sid, 'b s m -> b (s m)')
        
        # coords = self.reproject()[...,self.P//2, self.P//2]
        # pdb.set_trace()
      
        coords_init = None
        if S > 1 and self.is_initialized :
            # default: copy xys
            N = xys.shape[1]

            if self.cfg.slam.TRAJ_INIT == 'copy':
                coords_init = xys.clone().reshape(B, 1, N, 2).repeat(1, S, 1, 1)
            
            elif self.cfg.slam.TRAJ_INIT == 'reproj':
                # init from reprojection
                ii = []
                jj = []
                kk = []
                for s in range(S-1):
                    patch_ii = torch.ones(self.M * (S-1)) * (self.n-S+s)
                    patch_jj = repeat(torch.arange(S-1) + self.n-S, 's -> (m s)', m=self.M)
                    patch_kk = repeat(torch.arange(self.M) + (self.n-S+s) * self.M, 'm -> (m s)', s=S-1)
                    ii.append(patch_ii)
                    jj.append(patch_jj)
                    kk.append(patch_kk)
                    
                ii = torch.cat(ii).long()
                jj = torch.cat(jj).long()
                kk = torch.cat(kk).long()
                coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
                
                coords = rearrange(coords, 'b (s2 m s1) p1 p2 c -> b s1 s2 m (p1 p2 c)', s1=S-1, s2=S-1, p1=1, p2=1)
                coords_init = rearrange(coords_init, 'b s1 (s2 m) c -> b s1 s2 m c', s2=S, m=self.M)
                patch_valids = repeat(self.patches_valid_[self.n-S: self.n-1], 's2 m -> b s1 s2 m c', b=B, s1=S-1, c=2).bool()
                coords_init[:,:S-1,:S-1][patch_valids] = coords[patch_valids]
                coords_init = rearrange(coords_init, 'b s1 s2 m c -> b s1 (s2 m) c')
                
            elif self.cfg.slam.TRAJ_INIT in ['flow', 'gt']:
                last_target = self.last_target if self.cfg.slam.TRAJ_INIT == 'flow' else self.last_target_gt
                last_valid = self.last_valid if self.cfg.slam.TRAJ_INIT == 'flow' else self.last_valid_gt
                
                last_target = rearrange(self.last_target, 'b s1 (s2 m) c -> b s1 s2 m c', s2=S, m=self.M)
                last_valid = rearrange(self.last_valid, 'b s1 (s2 m) -> b s1 s2 m', s2=S, m=self.M)
                
                padding = 20
                boundary_mask = (last_target[...,0] >= padding) & (last_target[...,0] < self.wd - padding) & (last_target[...,1] >= padding) & (last_target[...,1] < self.ht - padding) 
                last_valid = last_valid & boundary_mask.to(last_valid.device)
                
                coords_init = rearrange(coords_init, 'b s1 (s2 m) c -> b s1 s2 m c', s2=S, m=self.M)
                last_valid = last_valid[:,1:, 1:]
                coords_init[:,:S-1,:S-1][last_valid] = last_target[:,1:,1:][last_valid]
                coords_init = rearrange(coords_init, 'b s1 s2 m c -> b s1 (s2 m) c')
 
        return xys, xys_sid, coords_init
    
    def _compute_sparse_tracks(
        self,
        video,
        queries,
    ):
        B, T, C, H, W = video.shape
        assert B == 1
        video = video.reshape(B * T, C, H, W).float()
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear")     # self.interp_shape = (384, 512)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        queries = queries.clone()
        B, N, D = queries.shape
        assert D == 3
        # scale query position according to interp_shape
        queries[:, :, 1] *= self.interp_shape[1] / W
        queries[:, :, 2] *= self.interp_shape[0] / H

        # xys_sid = queries[...,0]
        # xys = queries[...,1:]
        # tracks, visibilities, stats = self.network(rgbs=video, xys=xys, xys_sid=xys_sid, iters=self.cfg.model.I)
        stats = {}
        # start_time = time.time()
        with Timer("LEAP front-end", enabled=True):
            if self.cfg.model.mode in ['cotracker_new', 'cotracker_pips', 'cotracker_long']:
                tracks, _, visibilities, dynamic_e, _ = self.network(rgbs=video, queries=queries, iters=self.cfg.model.I)
                stats['dynamic_e'] = dynamic_e
            elif self.cfg.model.mode == 'cotracker_iid':
                tracks, _, visibilities, var_e, dynamic_e, _ = self.network(rgbs=video, queries=queries, iters=self.cfg.model.I)
                stats['var_e'] = var_e
                stats['dynamic_e'] = dynamic_e
            elif self.cfg.model.mode == 'cotracker_kernel_v2':
                tracks, _, visibilities, cov_list_e, dynamic_e, _ = self.network(rgbs=video, queries=queries, iters=self.cfg.model.I)
                stats['var_e'] = cov_list_e[0] + cov_list_e[1]  # var_x + var_y
                stats['dynamic_e'] = dynamic_e
        
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"leap front-end {elapsed_time} s")
            
        # TODO: open backward_tracking
        if self.cfg.slam.backward_tracking:
            tracks, visibilities, stats = self._compute_backward_tracks(
                video, queries, tracks, visibilities, stats
            )

        # TODO: batchify
        for i in range(len(queries)):
            queries_t = queries[i, :tracks.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))

            # overwrite the predictions with the query points
            tracks[i, queries_t, arange] = queries[i, :tracks.size(2), 1:]

            # correct visibilities, the query points should be visible
            visibilities[i, queries_t, arange] = 1.0

        tracks[:, :, :, 0] *= W / float(self.interp_shape[1])
        tracks[:, :, :, 1] *= H / float(self.interp_shape[0])

        return tracks, visibilities, stats


    def _compute_backward_tracks(self, video, queries, tracks, visibilities, stats):
        inv_video = video.flip(1).clone()
        inv_queries = queries.clone()
        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

        # inv_tracks, inv_visibilities, inv_stats = self.model(
        #     rgbs=inv_video, xys=inv_queries[...,1:], xys_sid=inv_queries[...,0], iters=self.cfg.model.I
        # )
        inv_stats = {}

        if self.cfg.model.mode in ['cotracker_new', 'cotracker_pips', 'cotracker_long']:
            inv_traj_e, _, inv_vis_e, inv_dynamic_e, _ = self.network(rgbs=inv_video, queries=inv_queries, iters=self.cfg.model.I)
            inv_stats['dynamic_e'] = inv_dynamic_e
        elif self.cfg.model.mode == 'cotracker_iid':
            inv_traj_e, _, inv_vis_e, inv_var_e, inv_dynamic_e, _ = self.network(rgbs=inv_video, queries=inv_queries, iters=self.cfg.model.I)
            inv_stats['var_e'] = inv_var_e
            inv_stats['dynamic_e'] = inv_dynamic_e
        elif self.cfg.model.mode == 'cotracker_kernel_v2':
            inv_traj_e, _, inv_vis_e, inv_cov_list_e, inv_dynamic_e, _ = self.network(rgbs=inv_video, queries=inv_queries, iters=self.cfg.model.I)
            inv_stats['var_e'] = inv_cov_list_e[0] + inv_cov_list_e[1]  # var_x + var_y
            inv_stats['dynamic_e'] = inv_dynamic_e

        inv_tracks = inv_traj_e.flip(1)
        inv_visibilities = inv_vis_e.flip(1)

        mask = tracks == 0

        tracks[mask] = inv_tracks[mask]
        visibilities[mask[:, :, :, 0]] = inv_visibilities[mask[:, :, :, 0]]
        for key, value in stats.items():
            if key in ['dynamic_e', 'var_e']:
                stats[key][mask[:, :, :, 0]] = inv_stats[key][mask[:, :, :, 0]]

        return tracks, visibilities, stats

    def get_window_trajs(self, only_coords=False):
        rgbs = torch.stack(self.local_window, dim=0).unsqueeze(0)   # B, S, C, H, W
        B, S_local, _, H, W = rgbs.shape

        queries = self.get_queries()
        
        # if only_coords:
        #     return xys, xys_sid
        
        # pad repeated frames to make local window = S
        if rgbs.shape[1] < self.S_slam:
            repeat_rgbs = repeat(rgbs[:,-1], 'b c h w -> b s c h w', s=self.S-S_local)
            rgbs = torch.cat([rgbs, repeat_rgbs], dim=1)
        
        static_label = None
        coords_vars = None
        conf_label = None

        if self.cfg.model.mode in ['cotracker_new', 'cotracker_iid', 'cotracker_kernel_v2', 'cotracker_pips', 'cotracker_long']:
            
          
            traj_e, vis_e, stats = self._compute_sparse_tracks(video=rgbs, queries=queries)
            local_target = traj_e
            if 'VIS_THRESHOLD' in self.cfg.slam:
                vis_label = (vis_e > self.cfg.slam.VIS_THRESHOLD)   # B, S, N
            else:
               vis_label = (torch.ones_like(vis_e) > 0)

            if 'dynamic_e' in stats and 'STATIC_THRESHOLD' in self.cfg.slam:
                # TODO: threshold should be calculated from each frame
                if self.cfg.model.mode == 'cotracker_long':
                    dynamic_e = torch.mean(stats['dynamic_e'], dim=1).unsqueeze(1).repeat(1, vis_label.shape[1], 1)
                    statie_e  = 1 - dynamic_e
                else:
                    statie_e = 1 - stats['dynamic_e']
                static_th = torch.quantile(statie_e,  (1 - self.cfg.slam.STATIC_QUANTILE))
                static_th = min(static_th.item(), self.cfg.slam.STATIC_THRESHOLD)
                # static_th = self.cfg.slam.STATIC_THRESHOLD
                static_label = statie_e >= static_th
                # static_label = repeat(static_label, 'b n -> b s n', s=vis_label.shape[1])
                print(f"vis_label@{self.cfg.slam.VIS_THRESHOLD}: {vis_label.float().mean().item():.4f} , static_label@{static_th}: {static_label.float().mean().item():.4f}")
                vis_label = vis_label & static_label

            if 'var_e' in stats and 'CONF_THRESHOLD' in self.cfg.slam:
                 # TODO: threshold should be calculated from each frame
                # coords_vars = stats['var_e']
                coords_vars = torch.sqrt(stats['var_e'])

                conf_th = torch.quantile(coords_vars,  self.cfg.slam.CONF_QUANTILE, dim=2, keepdim=True)
                # conf_th = torch.max(conf_th.item(), self.cfg.slam.CONF_THRESHOLD)
                conf_th[conf_th < self.cfg.slam.CONF_THRESHOLD] = self.cfg.slam.CONF_THRESHOLD
                # conf_th = self.cfg.slam.CONF_THRESHOLD
                conf_label = coords_vars < conf_th
                vis_label = vis_label & conf_label
                print(f"conf_label@{conf_th.mean()}: {conf_label.float().mean().item():.4f}, valid_label: {vis_label.float().mean().item():.4f}")

        elif self.cfg.model.mode in 'cotracker':
            xys_sid = queries[...,0]
            xys = queries[...,1:]
            local_target, vis_e = self.network(xys, rgbs, xys_sid=xys_sid, iters=self.cfg.model.I, coords_init=None)
            vis_e = vis_e.float()

            if 'VIS_THRESHOLD' in self.cfg.slam:
                vis_label = (vis_e > self.cfg.slam.VIS_THRESHOLD)   # B, S, N
            else:
                vis_label = (torch.ones_like(vis_e) > 0)

        local_target = local_target[:,:S_local]
        vis_label = vis_label[:,:S_local]

        # update patches valid
        if self.is_initialized:
            query_valid = self.patches_valid_[self.n-len(self.local_window):self.n:self.kf_stride]
            valid_from_filter = (vis_label.sum(dim=1) > 3) 
            query_valid = torch.logical_or(query_valid.reshape(-1), valid_from_filter)
            self.patches_valid_[self.n-len(self.local_window):self.n:self.kf_stride] = query_valid.reshape(-1, self.M)

        print("vis_label", vis_label.float().mean())
        stats = {
            'vis_label': None,
            'static_label': None,
            'conf_label': None,
            'coords_vars': None
        }

        if vis_label is not None: stats['vis_label'] = vis_label[:,:S_local]
        if static_label is not None: stats['static_label'] = static_label[:,:S_local]
        if conf_label is not None: stats['conf_label'] = conf_label[:,:S_local]
        if coords_vars is not None: stats['coords_vars'] = coords_vars[:,:S_local]
        
        return local_target, vis_label, queries, stats
    

    def add_traj_noise(self, trajs, cfg):
        mode, std = cfg.split('_')
        std = float(std)
        
        if mode == 'gauss':
            noise = torch.randn(*trajs.shape).to(trajs.device) * std
        elif mode == 'uniform':
            noise = (torch.rand_like(trajs) - 0.5) * 2 * std
        else:
            raise NotImplementedError
        
        trajs = trajs + noise
        return trajs
        
    def predict_target(self):
        # predict target        
        with torch.no_grad():
            if self.cfg.use_gt_traj:
                xys, xys_sid = self.get_window_trajs(only_coords=True)
            else:
                trajs, vis_label, queries, stats, = self.get_window_trajs()
                xys_sid = queries[...,0]
                xys = queries[...,1:]
            
        if self.cfg.load_gt:
            trajs_gt, valid_gt = self.get_gt_trajs(xys, xys_sid)
        else:
            trajs_gt = None
            valid_gt = None
             
        if self.cfg.use_gt_traj:
            trajs = trajs_gt
            vis_label = torch.ones(trajs_gt.shape[:3]).bool()

            if 'gt_traj_noise' in self.cfg:
                trajs = self.add_traj_noise(trajs, self.cfg.gt_traj_noise)
            
        # save predictions
        self.last_target = trajs
        self.last_valid = vis_label 
        
        if self.cfg.load_gt:
            self.last_target_gt = trajs_gt
            self.last_valid_gt = torch.all(valid_gt > 0, dim=-1)
        
        # rearrange s.t. it matches the edge order
        B, S, N, C = trajs.shape
        print("trajs", trajs.shape)
        local_target = rearrange(trajs, 'b s n c -> b (n s) c')
        
        # predict weight 
        if self.cfg.load_gt:
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
            edge_weights = repeat(edge_weight_decay_cum, 's -> b (n s) c', b=B, c=C, n=N)
            local_weight = local_weight * edge_weights

        # compute ransac masks
        if self.cfg.slam.USE_RANSAC:
            ransac_mask = self.compute_ransac_mask(trajs, mode=self.cfg.slam.RANSAC_MODE)
            ransac_mask = repeat(ransac_mask, 'b s n -> b (n s) c', c=C)
            local_weight[~ransac_mask] = 0
            
        # for target output boundary, set weight to 0
        padding = 20
        boundary_mask = (local_target[...,0] >= padding) & (local_target[...,0] < self.wd - padding) & (local_target[...,1] >= padding) & (local_target[...,1] < self.ht - padding) 
        local_weight[~boundary_mask] = 0 
        
        # GT FILTERING: using GT traj to filter pred
        if self.cfg.load_gt and 'GT_FILTERING' in self.cfg.slam and self.cfg.slam.GT_FILTERING > 0:
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
            # FIXME: check the query frame idx to replace
            self.patches_valid_[self.n-S:self.n:self.kf_stride] = patch_valid.reshape(-1, self.M)
            track_len_mask = repeat(patch_valid, 'b n -> b (n s)', s=S)
            local_weight[~track_len_mask] = 0

        # only keep matches with frame distance < threshold of reference points
        if 'MAX_FRAME_DIST' in self.cfg.slam and self.cfg.slam.MAX_FRAME_DIST:
            S = len(self.local_window)
            xys_sid = repeat(torch.arange(S).to(local_weight.device), 's -> b (s m)', b=B, m=self.M)
            xys_sid = repeat(xys_sid, 'b n -> b s n', s=S)
            frame_id = repeat(torch.arange(S).to(local_weight.device), 's -> b s n', b=B, n=N)
            frame_dist_mask = torch.abs(xys_sid - frame_id) <= self.cfg.slam.MAX_FRAME_DIST
            frame_dist_mask = rearrange(frame_dist_mask, 'b s n -> b (n s)')
            local_weight[~frame_dist_mask] = 0

        # append to global targets, weights
        self.targets = torch.cat([self.targets, local_target], dim=1)
        self.weights = torch.cat([self.weights, local_weight], dim=1)

        print("local_target", local_target.shape, "targets", self.targets.shape)
        local_target_ = rearrange(local_target, 'b (s1 m s) c -> b s s1 m c', s=S, m=self.M)
        local_weight_ = rearrange(local_weight, 'b (s1 m s) c -> b s s1 m c', s=S, m=self.M)
     
        # visaulization
        vis_data = {
            'fid': self.n,
            'targets': local_target_,
            'weights': local_weight_,
            'queries': queries
        }
        for key, value in stats.items():
            if value is not None:
                # value = rearrange(value, 'b s (s1 m) -> b s s1 m ', s=S, m=self.M)
                vis_data[key] = value


        self.visualizer.add_track(vis_data)
    
        # evaluate 
        if self.cfg.load_gt:
            valid_gt = (valid_gt).any(dim=-1)
            valid_pred = rearrange((local_weight > 0).any(dim=-1), ' b (n s) -> b s n', n=N, s=S)
            
            masked_ate = self.eval_ate(trajs, trajs_gt, valid_pred, valid_gt)
            
            if self.save_pips and self.n > 1 and masked_ate > self.cfg.save_pips_min_ate:
                rgbs = torch.stack(self.local_window, dim=0).unsqueeze(0) 
                if torch.isnan(trajs_gt).any():
                    pdb.set_trace()
                save_pips_path = os.path.join(self.save_dir,'saved_images')
                Path(save_pips_path).mkdir(exist_ok=True)
                # valid = valid_gt & valid_pred
                # masked_ate = torch.linalg.norm(trajs[0][valid[0]] - trajs_gt[0][valid[0]], dim=-1).mean()
                # print("masked_ate", masked_ate)
                # if masked_ate > 
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
      
    def compute_ransac_mask(self, trajs, mode='back'):
        """Compute the ransac mask based on trajectories
        
        Args:
            trajs: B, S, N, C
        Returns:    
            masks: B, S, N
        """
        # FIXME: Need to change to N vs N
        
        B, S, N, C = trajs.shape
        assert B==1
        # It is the maximum distance from a point to an epipolar line in pixels
        thresh = self.cfg.slam.RANSAC_THRESH if 'RANSAC_THRESH' in self.cfg.slam else 1.0
        
        trajs_np = trajs.detach().cpu().numpy()
        intrinsics = self.intrinsics[:,max(0,self.n-self.S):self.n].detach().cpu().numpy()
        Ks = np.zeros((B, S, 3, 3))
        Ks[:,:,0,0] = intrinsics[...,0]
        Ks[:,:,1,1] = intrinsics[...,1]
        Ks[:,:,0,2] = intrinsics[...,2]
        Ks[:,:,1,2] = intrinsics[...,3]
        Ks[:,:,2,2] = 1
        
        masks = np.ones((B, S, N)).astype(bool)
        if mode in ['forw', 'back']:
            if mode == 'back':
                ref_idx = S - 1
            elif mode == 'forw':
                ref_idx = 0
            for i in range(S):
                pts1 = trajs_np[0,i]
                pts2 = trajs_np[0, ref_idx]
                mask = ransac_mask(pts1, pts2, Ks[0,i], Ks[0,ref_idx], ransac=True, thresh=thresh)
                masks[0,i] = mask.reshape(-1)
        elif mode == 'neighbor':
            for i in range(S-1):
                pts1 = trajs_np[0,i]
                pts2 = trajs_np[0, i+1]
                mask = ransac_mask(pts1, pts2, Ks[0,i], Ks[0,i+1], ransac=True, thresh=thresh)
                masks[0,i] = masks[0,i] & mask
                masks[0,i+1] = masks[0,i+1] & mask


        masks = torch.from_numpy(masks).to(trajs.device).bool()
        return masks
    
    def update(self):
        # lmbda
        lmbda = torch.as_tensor([1e-4], device="cuda")
        
        # ba
        t0 = self.n - self.cfg.slam.OPTIMIZATION_WINDOW if self.is_initialized else 1
        t0 = max(t0, 1)

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
        tstamps = np.array(self.tlist, dtype=float)

        if self.viewer is not None:
            self.viewer.join()

        # save metrics
        save_metrics_path = os.path.join(self.save_dir,'saved_metrics')
        Path(save_metrics_path).mkdir(exist_ok=True)
        for name, metric_list in self.metrics.items():
            np.save(f'{save_metrics_path}/{name}', np.array(metric_list))

        return poses, tstamps
    
    
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
        if self.visualizer is not None:
            self.visualizer.add_frame(image)

        # image preprocessing   
        self.preprocess(image, intrinsics)
        
        # for debug
        if depth_g is not None and cam_g is not None:
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

        if self.n % self.kf_stride == 0 and not self.is_initialized:  
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
            
        self.index_map_[self.n] = self.m
        
        self.counter += 1    

        self.n += 1
        self.m += self.M

        if (self.n - 1) % self.kf_stride == 0:  
            self.append_factors(*self.__edges())
            self.predict_target()

        if self.n == self.cfg.slam.num_init and not self.is_initialized:
            print("initialized !")
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
        if self.cfg.load_gt:
            self.print_stats()

        # if self.cfg.save_results:
        #     self.save_results()

        torch.cuda.empty_cache()

    def keyframe(self):
        to_remove = self.ix[self.kk] < self.n - self.cfg.slam.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def print_stats(self):
        line = "[metrics]"
        for name, metric_list in self.metrics.items():
            # line += f" {name}: {np.mean(metric_list)}"
            line += f" {name}: {metric_list[-1]}"
        print(line)

        # print("invalid", self.invalid_frames)
    
    # def save_results(self):
    #     """Save the camera pose and 3D point at the moment
    #     """
    #     Gs = SE3(self.poses_).detach()
    #     pose = Gs.matrix()[:self.n].float().detach().cpu().numpy()
    #     pts = self.points_[:self.n * self.M].float().detach().cpu().numpy()
    #     clr = self.colors_[:self.n].reshape(-1,3).float().detach().cpu().numpy()
        
    #     pts_valid = self.patches_valid_[:self.n].reshape(-1).detach().cpu().numpy()

    #     save_results_path = os.path.join(self.save_dir,'saved_results')
    #     Path(save_results_path).mkdir(exist_ok=True)
    #     save_name = os.path.join(save_results_path, f'{self.n}.npz')
    #     np.savez(save_name, pose=pose, pts=pts, clr=clr, pts_valid=pts_valid)

    def get_results(self):
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().matrix().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=float)


        # Gs = SE3(self.poses_[:]).detach()
        # pose = Gs.matrix()[:self.n].float().detach().cpu().numpy()
        pts = self.points_[:self.counter * self.M].reshape(-1,self.M, 3).float().detach().cpu().numpy()
        clrs = self.colors_[:self.counter].float().detach().cpu().numpy()
        pts_valid = self.patches_valid_[:self.counter].detach().cpu().numpy()

        intrinsics = self.intrinsics_[:self.counter].detach().cpu().numpy()

        patches = self.patches_[:self.counter,:, :, self.P//2, self.P//2].detach().cpu().numpy()
        return poses, intrinsics, pts, clrs, pts_valid, patches, tstamps


    def save_results(self, save_path, imagedir):
        """Save the camera pose and 3D point at the moment
        """

        
        # save_results_path = os.path.join(self.save_dir,'saved_results.npz')
        # Path(save_path).mkdir(exist_ok=True)
        # save_name = os.path.join(save_results_path, f'{self.counter}.npz')
        # save_path = os.path.join(save_dir, save_name)
        poses, intrinsics, pts, clrs, pts_valid, patches, tstamps = self.get_results()
        np.savez(
            save_path, 
            imagedir=imagedir,
            poses=poses, 
            intrinsics=intrinsics,
            pts=pts, 
            clrs=clrs, 
            pts_valid=pts_valid, 
            patches=patches,
            tstamps=tstamps
        )
        print("save results", save_path, self.counter)



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


@hydra.main(version_base=None, config_path="configs", config_name="pipsmultislam")
def main(cfg: DictConfig):

    gt_traj = load_traj(cfg.data.gt_traj, cfg.data.traj_format, skip=cfg.data.skip, stride=cfg.data.stride)

    slam = None
    timeit = False
    skip = 0
    
    # queue = Queue(maxsize=1)


    imagedir, calib, stride, skip = cfg.data.imagedir, cfg.data.calib, cfg.data.stride, cfg.data.skip 
    print("Running with config...")
    print(cfg)
    print(imagedir, cfg.data.name)

    # if os.path.isdir(cfg.data.imagedir):
    #     reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    # else:
    #     reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    # reader.start()
    # if os.path.isdir(cfg.data.imagedir):
    #     dataloader = image_stream(None, imagedir, calib, stride, skip)
    # else:
    #     dataloader = video_stream(None, imagedir, calib, stride, skip)
    # dataloader = replica_stream(imagedir, calib, stride, skip)
    if cfg.data.traj_format == 'sintel':
        dataloader = sintel_stream(None, imagedir, calib, stride, skip)
    else:
        dataloader = dataset_stream(None, imagedir, calib, stride, skip, mode=cfg.data.traj_format)
    # gt_traj = load_gt_traj(cfg.data.gt_traj, skip=cfg.data.skip, stride=cfg.data.stride, traj_format=cfg.data.traj_format)
    
    image_list = []
    intrinsics_list = []
    for i, (t, image, intrinsics) in enumerate(dataloader):

        if "max_length" in cfg.data and i >= cfg.data.max_length: break
        if t < 0: break
        
        print(i, image.shape, intrinsics.shape)
        
        image_list.append(image)
        intrinsics_list.append(intrinsics)
        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()
        
        
        # initialization
        if slam is None:
            slam = COTRACKERSLAM(cfg, ht=image.shape[1], wd=image.shape[2])
        # pdb.set_trace()
        # tracking

        with Timer("SLAM", enabled=True):
            slam(t, image, intrinsics, depth_g=None, cam_g=None)

    # tuple (N, 7), (N, s)
    pred_traj = slam.terminate()

    if cfg.data.traj_format == 'tum':
        traj_t_map_file = cfg.data.gt_traj.replace('groundtruth.txt', 'rgb.txt')
        pred_traj = list(pred_traj)
        pred_traj[1] = load_timestamps(traj_t_map_file, cfg.data.traj_format)
        pred_traj[1] = pred_traj[1][:pred_traj[0].shape[0]]
    elif cfg.data.traj_format == 'tartan_shibuya':
        traj_t_map_file = cfg.data.gt_traj.replace('gt_pose.txt', 'times.txt')
        pred_traj = list(pred_traj)
        pred_traj[1] = load_timestamps(traj_t_map_file, cfg.data.traj_format)
        pred_traj[1] = pred_traj[1][:pred_traj[0].shape[0]]
    # elif args.traj_format == 'sintel':
    

    os.makedirs(f"{cfg.data.savedir}/{cfg.data.name}", exist_ok=True)

    if cfg.save_results:
        save_results_path  = f"{cfg.data.savedir}/{cfg.data.name}/saved_results.npz"
        slam.save_results(save_results_path, imagedir=cfg.data.imagedir)

    if cfg.save_trajectory:
        # Path(os.path.join(slam.save_dir, "saved_trajectories")).mkdir(exist_ok=True)
        # save_trajectory_tum_format(pred_traj, f"saved_trajectories/{cfg.exp_name}.txt")
        # save_trajectory_tum_format(pred_traj, os.path.join(slam.save_dir,'traj.txt'))
        save_trajectory_tum_format(pred_traj, f"{cfg.data.savedir}/{cfg.data.name}/pipsmultislam_traj.txt")

    if cfg.plot:
        # Path("trajectory_plots").mkdir(exist_ok=True)
        # plot_trajectory(pred_traj, gt_traj=gt_traj, title=f"PIP_SLAM Trajectory Prediction for {cfg.exp_name}", filename=os.path.join(slam.save_dir,'traj_plot.pdf'))
        plot_trajectory(pred_traj, gt_traj=gt_traj, title=f"DPVO Trajectory Prediction for {cfg.exp_name}", filename=f"{cfg.data.savedir}/{cfg.data.name}/traj_plot.pdf")
    
    if cfg.save_video:
        slam.visualizer.save_video(filename=cfg.slam.PATCH_GEN)

    # eval_metrics(pred_traj, gt_traj=gt_traj, seq=cfg.exp_name, filename=os.path.join(slam.save_dir,'eval_metrics.txt'))
    ate, rpe_trans, rpe_rot = eval_metrics(pred_traj, gt_traj=gt_traj, seq=cfg.exp_name, filename=os.path.join(cfg.data.savedir,cfg.data.name, 'eval_metrics.txt'))
    with open(os.path.join(cfg.data.savedir, 'error_sum.txt'), 'a+') as f:
        line = f"{cfg.data.name:<20} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
        f.write(line)
        line = f"{ate:.5f}\n{rpe_trans:.5f}\n{rpe_rot:.5f}\n"
        f.write(line)


    # visualization
    if True:
        vis_rerun(slam, image_list, intrinsics_list)


if __name__ == '__main__':
    main()