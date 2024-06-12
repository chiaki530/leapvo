import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from main.backend import altcorr, lietorch
from main.backend import projective_ops as pops
from main.backend.ba import BA
from main.backend.lietorch import SE3
from main.leap.leap_kernel import LeapKernel
from main.slam_visualizer import LEAPVisualizer


def flatmeshgrid(*args, **kwargs):
    grid = torch.meshgrid(*args, **kwargs)
    return (x.reshape(-1) for x in grid)


def coords_grid_with_index(d, **kwargs):
    """coordinate grid with frame index"""
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


class LEAPVO:
    def __init__(self, cfg, ht=480, wd=640):

        self.cfg = cfg
        self.load_weights()
        self.ht = ht
        self.wd = wd
        self.P = 1  # point tracking: patch_size = 1
        self.S = cfg.model.S
        self.is_initialized = False
        self.enable_timing = False
        self.pred_back = cfg.pred_back if "pred_back" in cfg else None

        self.n = 0  # number of frames
        self.m = 0  # number of patches
        self.M = self.cfg.slam.PATCHES_PER_FRAME
        self.N = self.cfg.slam.BUFFER_SIZE

        # dummy image for visualization
        self.tlist = []
        self.counter = 0

        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(
            self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda"
        )
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.patches_valid_ = torch.zeros(
            self.N, self.M, dtype=torch.float, device="cuda"
        )
        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.targets = torch.zeros(1, 0, 2, device="cuda")
        self.weights = torch.zeros(1, 0, 2, device="cuda")

        # initialize poses to identity matrix
        self.poses_[:, 6] = 1.0

        self.local_window = []

        # store relative poses for removed frames
        self.delta = {}

        self.viewer = None

        # cache
        self.cache_window = []
        self.invalid_frames = []

        self.S_model = cfg.model.S
        self.S_slam = cfg.slam.S_slam
        self.S = cfg.slam.S_slam
        self.kf_stride = cfg.slam.kf_stride
        self.interp_shape = (384, 512)

        save_dir = f"{cfg.data.savedir}/{cfg.data.name}"

        self.use_forward = cfg.slam.use_forward if "use_forward" in cfg.slam else True
        self.use_backward = (
            cfg.slam.use_backward if "use_backward" in cfg.slam else True
        )

        self.visualizer = LEAPVisualizer(cfg, save_dir=save_dir)

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N * self.M, 3, self.P, self.P)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    def init_motion(self):
        if self.n > 1:
            if self.cfg.slam.MOTION_MODEL == "DAMPED_LINEAR":
                P1 = SE3(self.poses_[self.n - 1])
                P2 = SE3(self.poses_[self.n - 2])

                xi = self.cfg.slam.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n - 1]
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
        self.ii = torch.cat([self.ii, self.ix[ii]])

    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.targets = self.targets[:, ~m]
        self.weights = self.weights[:, ~m]

    def __image_gradient_2(self, images):
        images_pad = F.pad(images, (1, 1, 1, 1), "constant", 0)
        gray = images_pad.sum(dim=2)
        dx = gray[..., :-1, 1:] - gray[..., :-1, :-1]
        dy = gray[..., 1:, :-1] - gray[..., :-1, :-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def generate_patches(self, image):
        B = 1

        if self.cfg.slam.PATCH_GEN == "random":
            x = torch.randint(1, self.wd - 1, size=[1, self.M], device="cuda")
            y = torch.randint(1, self.ht - 1, size=[1, self.M], device="cuda")
            coords = torch.stack([x, y], dim=-1).float()

        elif "grid_grad" in self.cfg.slam.PATCH_GEN:
            rel_margin = 0.15
            num_expand = 8

            grid_size = int(self.cfg.slam.PATCH_GEN.split("_")[-1])
            num_grid = grid_size * grid_size
            grid_M = self.M // num_grid
            H_grid, W_grid = self.ht // grid_size, self.wd // grid_size

            g = self.__image_gradient_2(self.local_window[-1][None, None, ...])

            x = (
                torch.rand((num_grid, num_expand * grid_M), device="cuda")
                * (1 - 2 * rel_margin)
                + rel_margin
            )
            y = (
                torch.rand((num_grid, num_expand * grid_M), device="cuda")
                * (1 - 2 * rel_margin)
                + rel_margin
            )
            # map to coordinate
            offset = torch.linspace(0, grid_size - 1, grid_size)
            offset_y, offset_x = torch.meshgrid(offset, offset)
            offset = torch.stack([offset_x, offset_y], dim=-1).to("cuda")
            offset = offset.view(-1, 2)
            offset[..., 0] = offset[..., 0] * W_grid
            offset[..., 1] = offset[..., 1] * H_grid

            x_global = x.view(1, num_grid, -1) * W_grid + offset[..., 0].view(1, -1, 1)
            y_global = y.view(1, num_grid, -1) * H_grid + offset[..., 1].view(1, -1, 1)

            coords = torch.stack([x_global, y_global], dim=-1).float()  ## [1, N, 2]
            coords = rearrange(coords, "b g n c -> b (g n) c")
            coords = torch.round(coords).unsqueeze(1)
            coords_norm = coords
            coords_norm[..., 0] = coords_norm[..., 0] / (self.wd - 1) * 2.0 - 1.0
            coords_norm[..., 1] = coords_norm[..., 0] / (self.ht - 1) * 2.0 - 1.0

            gg = F.grid_sample(g, coords_norm, mode="bilinear", align_corners=True)
            gg = gg[:, 0, 0]
            gg = rearrange(gg, "b (ng n) -> b ng n", ng=num_grid)
            ix = torch.argsort(gg, dim=-1)
            x_global = torch.gather(x_global, 2, ix[:, :, -grid_M:])
            y_global = torch.gather(y_global, 2, ix[:, :, -grid_M:])
            coords = torch.concat([x_global, y_global], dim=-1).float()

        disps = torch.ones(B, 1, self.ht, self.wd, device="cuda")
        grid, _ = coords_grid_with_index(disps, device=self.poses_.device)
        patches = altcorr.patchify(grid[0], coords, self.P // 2).view(
            B, -1, 3, self.P, self.P
        )  # B, N, 3, p, p

        clr = altcorr.patchify(image.unsqueeze(0).float(), (coords + 0.5), 0).view(
            B, -1, 3
        )

        return patches, clr

    def map_point_filtering(self):
        coords = self.reproject()[..., self.P // 2, self.P // 2]
        ate = torch.norm(coords - self.targets, dim=-1)
        reproj_mask = ate < self.cfg.slam.MAP_FILTERING_TH
        self.weights[~reproj_mask] = 0

    def reproject(self, indicies=None):
        """reproject patch k from i -> j"""
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords = pops.transform(
            SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk
        )
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def load(self):
        strict = True
        if self.cfg.model.init_dir != "":
            state_dict = torch.load(self.cfg.model.init_dir, map_location="cuda:0")
            if "model" in state_dict:
                state_dict = state_dict["model"]

            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            self.network.load_state_dict(state_dict, strict=strict)

    def load_weights(self):
        if self.cfg.model.mode == "leap_kernel":
            self.network = LeapKernel(cfg=self.cfg, stride=self.cfg.model.stride).cuda()
            self.load()
            self.network.eval()
        else:
            raise NotImplementedError

    def preprocess(self, image, intrinsics):
        """Load the image and store in the local window"""
        if len(self.local_window) >= self.S:
            self.local_window.pop(0)
        self.local_window.append(image)

        self.intrinsics_[self.n] = intrinsics

        torch.cuda.empty_cache()

    def __edges(self):
        """Edge between keyframe patches and the all local frames"""
        r = self.cfg.slam.S_slam
        local_start_fid = max((self.n - r), 0)
        local_end_fid = max((self.n - 0), 0)
        idx = torch.arange(0, self.n * self.M, device="cuda").reshape(self.n, self.M)
        kf_idx = idx[local_start_fid : local_end_fid : self.kf_stride].reshape(-1)

        return flatmeshgrid(
            kf_idx,
            torch.arange(max(self.n - self.S_slam, 0), self.n, device="cuda"),
            indexing="ij",
        )

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

        depths = (
            torch.stack(self.local_window_depth_g, dim=0).unsqueeze(0).to(xys.device)
        )  # B, S, C, H, W
        cams_c2w = (
            torch.stack(self.local_window_cam_g, dim=0).unsqueeze(0).to(xys.device)
        )  # B, S, C, H, W
        intrinsics = self.intrinsics[:, self.n - S : self.n].to(xys.device)

        assert len(self.local_window_cam_g) == len(self.local_window_depth_g)

        # back-project xy from each frame
        P0 = torch.empty(B, N, 4).to(xys.device)
        xy_depth = torch.empty(B, N, 1).to(xys.device)
        for s in range(S):
            mask = xys_sid == s
            xys_s = xys[mask].reshape(B, self.M, 2)
            depth_s = altcorr.patchify(depths[:, [s]].float(), xys_s, 0).reshape(
                B, self.M, 1
            )
            xy_depth[mask] = depth_s.reshape(-1, 1)
            P0[mask] = pops.back_proj(
                xys_s, depth_s, intrinsics[:, s], cams_c2w[:, s]
            ).reshape(-1, 4)

        # project to all frame in the local window
        cams_w2c = torch.inverse(cams_c2w)
        xy_gt = pops.proj_to_frames(P0, intrinsics, cams_w2c)

        xy_gt = xy_gt[:, :S]

        # Detect NAN value
        xy_repeat = repeat(xys, "b n c -> b s n c", s=S)
        invalid = torch.isnan(xy_gt) | torch.isinf(xy_gt)
        invalid_depth = (xy_depth <= 0) | torch.isnan(xy_depth) | torch.isinf(xy_depth)
        invalid_depth = repeat(invalid_depth, "b n i -> b s n (i c)", s=S, c=2)
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
        xys = self.patches_[self.n - S : self.n, :, :2, self.P // 2, self.P // 2]
        xys = xys.unsqueeze(0)  # B, S, M, 2

        B = xys.shape[0]
        # compute xys_sid
        xys_sid = repeat(torch.arange(S).to(xys.device), "s -> b s m", b=B, m=self.M)

        xys = rearrange(xys[:, :: self.kf_stride], "b s m c -> b (s m) c")
        xys_sid = rearrange(xys_sid[:, :: self.kf_stride], "b s m -> b (s m)")

        queries = torch.cat([xys_sid.unsqueeze(-1), xys], dim=2)

        return queries

    def get_patches_xy(self):
        S = len(self.local_window)
        # extract the patches from local windows
        xys = self.patches_[
            self.n - S : self.n, :, :2, self.P // 2, self.P // 2
        ]  # S, M, 2
        xys = xys.unsqueeze(0)  # B, S, M, 2

        B = xys.shape[0]
        # compute xys_sid
        xys_sid = repeat(torch.arange(S).to(xys.device), "s -> b s m", b=B, m=self.M)
        xys = rearrange(xys, "b s m c -> b (s m) c")
        xys_sid = rearrange(xys_sid, "b s m -> b (s m)")

        coords_init = None
        if S > 1 and self.is_initialized:
            N = xys.shape[1]

            if self.cfg.slam.TRAJ_INIT == "copy":
                coords_init = xys.clone().reshape(B, 1, N, 2).repeat(1, S, 1, 1)

            elif self.cfg.slam.TRAJ_INIT == "reproj":
                # init from reprojection
                ii = []
                jj = []
                kk = []
                for s in range(S - 1):
                    patch_ii = torch.ones(self.M * (S - 1)) * (self.n - S + s)
                    patch_jj = repeat(
                        torch.arange(S - 1) + self.n - S, "s -> (m s)", m=self.M
                    )
                    patch_kk = repeat(
                        torch.arange(self.M) + (self.n - S + s) * self.M,
                        "m -> (m s)",
                        s=S - 1,
                    )
                    ii.append(patch_ii)
                    jj.append(patch_jj)
                    kk.append(patch_kk)

                ii = torch.cat(ii).long()
                jj = torch.cat(jj).long()
                kk = torch.cat(kk).long()
                coords = pops.transform(
                    SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk
                )

                coords = rearrange(
                    coords,
                    "b (s2 m s1) p1 p2 c -> b s1 s2 m (p1 p2 c)",
                    s1=S - 1,
                    s2=S - 1,
                    p1=1,
                    p2=1,
                )
                coords_init = rearrange(
                    coords_init, "b s1 (s2 m) c -> b s1 s2 m c", s2=S, m=self.M
                )
                patch_valids = repeat(
                    self.patches_valid_[self.n - S : self.n - 1],
                    "s2 m -> b s1 s2 m c",
                    b=B,
                    s1=S - 1,
                    c=2,
                ).bool()
                coords_init[:, : S - 1, : S - 1][patch_valids] = coords[patch_valids]
                coords_init = rearrange(coords_init, "b s1 s2 m c -> b s1 (s2 m) c")

        return xys, xys_sid, coords_init

    def _compute_sparse_tracks(
        self,
        video,
        queries,
    ):
        B, T, C, H, W = video.shape
        assert B == 1
        video = video.reshape(B * T, C, H, W).float()
        video = F.interpolate(
            video, tuple(self.interp_shape), mode="bilinear"
        )  # self.interp_shape = (384, 512)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        queries = queries.clone()
        B, N, D = queries.shape
        assert D == 3
        # scale query position according to interp_shape
        queries[:, :, 1] *= self.interp_shape[1] / W
        queries[:, :, 2] *= self.interp_shape[0] / H

        stats = {}
        tracks, _, visibilities, cov_list_e, dynamic_e, _ = self.network(
            rgbs=video, queries=queries, iters=self.cfg.model.I
        )
        stats["var_e"] = cov_list_e[0] + cov_list_e[1]  # var_x + var_y
        stats["dynamic_e"] = dynamic_e

        # backward_tracking
        if self.cfg.slam.backward_tracking:
            tracks, visibilities, stats = self._compute_backward_tracks(
                video, queries, tracks, visibilities, stats
            )

        # TODO: batchify
        for i in range(len(queries)):
            queries_t = queries[i, : tracks.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))

            # overwrite the predictions with the query points
            tracks[i, queries_t, arange] = queries[i, : tracks.size(2), 1:]

            # correct visibilities, the query points should be visible
            visibilities[i, queries_t, arange] = 1.0

        tracks[:, :, :, 0] *= W / float(self.interp_shape[1])
        tracks[:, :, :, 1] *= H / float(self.interp_shape[0])

        return tracks, visibilities, stats

    def _compute_backward_tracks(self, video, queries, tracks, visibilities, stats):
        inv_video = video.flip(1).clone()
        inv_queries = queries.clone()
        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

        inv_stats = {}

        inv_traj_e, _, inv_vis_e, inv_cov_list_e, inv_dynamic_e, _ = self.network(
            rgbs=inv_video, queries=inv_queries, iters=self.cfg.model.I
        )
        inv_stats["var_e"] = inv_cov_list_e[0] + inv_cov_list_e[1]  # var_x + var_y
        inv_stats["dynamic_e"] = inv_dynamic_e

        inv_tracks = inv_traj_e.flip(1)
        inv_visibilities = inv_vis_e.flip(1)

        mask = tracks == 0

        tracks[mask] = inv_tracks[mask]
        visibilities[mask[:, :, :, 0]] = inv_visibilities[mask[:, :, :, 0]]
        for key, value in stats.items():
            if key in ["dynamic_e", "var_e"]:
                stats[key][mask[:, :, :, 0]] = inv_stats[key][mask[:, :, :, 0]]

        return tracks, visibilities, stats

    def get_window_trajs(self, only_coords=False):
        rgbs = torch.stack(self.local_window, dim=0).unsqueeze(0)  # B, S, C, H, W
        B, S_local, _, H, W = rgbs.shape

        queries = self.get_queries()

        # pad repeated frames to make local window = S
        if rgbs.shape[1] < self.S_slam:
            repeat_rgbs = repeat(
                rgbs[:, -1], "b c h w -> b s c h w", s=self.S - S_local
            )
            rgbs = torch.cat([rgbs, repeat_rgbs], dim=1)

        static_label = None
        coords_vars = None
        conf_label = None

        traj_e, vis_e, stats = self._compute_sparse_tracks(video=rgbs, queries=queries)
        local_target = traj_e
        if "VIS_THRESHOLD" in self.cfg.slam:
            vis_label = vis_e > self.cfg.slam.VIS_THRESHOLD  # B, S, N
        else:
            vis_label = torch.ones_like(vis_e) > 0

        if "dynamic_e" in stats and "STATIC_THRESHOLD" in self.cfg.slam:
            if self.cfg.model.mode == "cotracker_long":
                dynamic_e = (
                    torch.mean(stats["dynamic_e"], dim=1)
                    .unsqueeze(1)
                    .repeat(1, vis_label.shape[1], 1)
                )
                statie_e = 1 - dynamic_e
            else:
                statie_e = 1 - stats["dynamic_e"]
            static_th = torch.quantile(statie_e, (1 - self.cfg.slam.STATIC_QUANTILE))
            static_th = min(static_th.item(), self.cfg.slam.STATIC_THRESHOLD)
            static_label = statie_e >= static_th
            vis_label = vis_label & static_label

        if "var_e" in stats and "CONF_THRESHOLD" in self.cfg.slam:
            coords_vars = torch.sqrt(stats["var_e"])
            conf_th = torch.quantile(
                coords_vars, self.cfg.slam.CONF_QUANTILE, dim=2, keepdim=True
            )
            conf_th[conf_th < self.cfg.slam.CONF_THRESHOLD] = (
                self.cfg.slam.CONF_THRESHOLD
            )
            conf_label = coords_vars < conf_th
            vis_label = vis_label & conf_label

        local_target = local_target[:, :S_local]
        vis_label = vis_label[:, :S_local]

        # update patches valid
        if self.is_initialized:
            query_valid = self.patches_valid_[
                self.n - len(self.local_window) : self.n : self.kf_stride
            ]
            valid_from_filter = vis_label.sum(dim=1) > 3
            query_valid = torch.logical_or(
                query_valid.reshape(1, -1), valid_from_filter
            )
            self.patches_valid_[
                self.n - len(self.local_window) : self.n : self.kf_stride
            ] = query_valid.reshape(-1, self.M)

        stats = {
            "vis_label": None,
            "static_label": None,
            "conf_label": None,
            "coords_vars": None,
        }

        if vis_label is not None:
            stats["vis_label"] = vis_label[:, :S_local]
        if static_label is not None:
            stats["static_label"] = static_label[:, :S_local]
        if conf_label is not None:
            stats["conf_label"] = conf_label[:, :S_local]
        if coords_vars is not None:
            stats["coords_vars"] = coords_vars[:, :S_local]

        return local_target, vis_label, queries, stats

    def predict_target(self):
        # predict target
        with torch.no_grad():
            (
                trajs,
                vis_label,
                queries,
                stats,
            ) = self.get_window_trajs()

        # save predictions
        self.last_target = trajs
        self.last_valid = vis_label

        B, S, N, C = trajs.shape
        local_target = rearrange(trajs, "b s n c -> b (n s) c")

        # predict weight
        local_weight = torch.ones_like(local_target)

        vis_label = rearrange(vis_label, "b s n -> b (n s)")
        local_weight[~vis_label] = 0

        # out of boundary
        padding = 20
        boundary_mask = (
            (local_target[..., 0] >= padding)
            & (local_target[..., 0] < self.wd - padding)
            & (local_target[..., 1] >= padding)
            & (local_target[..., 1] < self.ht - padding)
        )
        local_weight[~boundary_mask] = 0

        # check track length
        if self.n >= self.cfg.slam.MIN_TRACK_LEN:
            patch_valid = (local_weight > 0).any(dim=-1)
            patch_valid = rearrange(patch_valid, "b (n s) -> b s n", s=S, n=N)
            patch_valid = patch_valid.sum(dim=1) >= self.cfg.slam.MIN_TRACK_LEN
            self.patches_valid_[self.n - S : self.n : self.kf_stride] = (
                patch_valid.reshape(-1, self.M)
            )
            track_len_mask = repeat(patch_valid, "b n -> b (n s)", s=S)
            local_weight[~track_len_mask] = 0

        # append to global targets, weights
        self.targets = torch.cat([self.targets, local_target], dim=1)
        self.weights = torch.cat([self.weights, local_weight], dim=1)

        local_target_ = rearrange(
            local_target, "b (s1 m s) c -> b s s1 m c", s=S, m=self.M
        )
        local_weight_ = rearrange(
            local_weight, "b (s1 m s) c -> b s s1 m c", s=S, m=self.M
        )

        # visaulization
        vis_data = {
            "fid": self.n,
            "targets": local_target_,
            "weights": local_weight_,
            "queries": queries,
        }
        for key, value in stats.items():
            if value is not None:
                vis_data[key] = value

        self.visualizer.add_track(vis_data)

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
            Gs, patches = BA(
                Gs,
                patches,
                self.intrinsics.detach(),
                self.targets.detach(),
                self.weights.detach(),
                lmbda,
                self.ii,
                self.jj,
                self.kk,
                bounds,
                ep=ep,
                fixedp=t0,
                structure_only=False,
                loss=self.cfg.slam.LOSS,
            )

        # for keeping the same memory -> viewer works
        self.patches_[:] = patches.reshape(self.N, self.M, 3, self.P, self.P)
        self.poses_[:] = Gs.vec().reshape(self.N, 7)

        # 3D points culling
        if self.cfg.slam.USE_MAP_FILTERING:
            with torch.no_grad():
                self.map_point_filtering()

        # TODO: debug extracting point
        points = pops.point_cloud(
            SE3(self.poses),
            self.patches[:, : self.m],
            self.intrinsics,
            self.ix[: self.m],
        )
        points = (
            points[..., self.P // 2, self.P // 2, :3]
            / points[..., self.P // 2, self.P // 2, 3:]
        ).reshape(-1, 3)
        self.points_[: len(points)] = points[:]

    def terminate(self):
        """interpolate missing poses"""
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=float)

        if self.viewer is not None:
            self.viewer.join()

        return poses, tstamps

    def __call__(self, tstamp, image, intrinsics):
        """main function of tracking

        Args:
            tstamp (_type_): _description_
            image (_type_): 3, H, W
            intrinsics (_type_): fx, fy, cx, cy

        Raises:
            Exception: _description_
        """
        if (self.n + 1) >= self.N:
            raise Exception(
                f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"'
            )

        if self.viewer is not None:
            self.viewer.update_image(image)
        if self.visualizer is not None:
            self.visualizer.add_frame(image)

        # image preprocessing
        self.preprocess(image, intrinsics)

        # generate patches
        patches, clr = self.generate_patches(image)

        # depth initialization
        patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])
        if self.is_initialized:
            s = torch.median(self.patches_[self.n - 3 : self.n, :, 2])
            patches[:, :, 2] = s

        self.patches_[self.n] = patches

        if self.n % self.kf_stride == 0 and not self.is_initialized:
            self.patches_valid_[self.n] = 1

        # pose initialization with motion model
        self.init_motion()

        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter

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
            self.is_initialized = True
            # one initialized, run global BA
            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            self.keyframe()

        torch.cuda.empty_cache()

    def keyframe(self):
        to_remove = self.ix[self.kk] < self.n - self.cfg.slam.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def get_results(self):
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().matrix().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=float)

        pts = (
            self.points_[: self.counter * self.M]
            .reshape(-1, self.M, 3)
            .float()
            .detach()
            .cpu()
            .numpy()
        )
        clrs = self.colors_[: self.counter].float().detach().cpu().numpy()
        pts_valid = self.patches_valid_[: self.counter].detach().cpu().numpy()

        intrinsics = self.intrinsics_[: self.counter].detach().cpu().numpy()

        patches = (
            self.patches_[: self.counter, :, :, self.P // 2, self.P // 2]
            .detach()
            .cpu()
            .numpy()
        )
        return poses, intrinsics, pts, clrs, pts_valid, patches, tstamps
