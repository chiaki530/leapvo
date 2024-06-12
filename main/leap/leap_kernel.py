import torch
import torch.nn as nn
from einops import rearrange, repeat
from leap.core.anchor_sampler import get_anchors
from leap.core.cotracker.blocks import (BasicEncoder, CorrBlock,
                                        MotionLabelBlock, UpdateFormer)
from leap.core.embeddings import (get_1d_sincos_pos_embed_from_grid,
                                  get_2d_embedding, get_2d_sincos_pos_embed)
from leap.core.model_utils import bilinear_sample2d, meshgrid2d, smart_cat

torch.manual_seed(0)


def get_points_on_a_grid(grid_size, interp_shape, grid_center=(0, 0), device="cpu"):
    if grid_size == 1:
        return torch.tensor([interp_shape[1] / 2, interp_shape[0] / 2], device=device)[
            None, None
        ]

    grid_y, grid_x = meshgrid2d(
        1, grid_size, grid_size, stack=False, norm=False, device=device
    )
    step = interp_shape[1] // 64
    if grid_center[0] != 0 or grid_center[1] != 0:
        grid_y = grid_y - grid_size / 2.0
        grid_x = grid_x - grid_size / 2.0
    grid_y = step + grid_y.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[0] - step * 2
    )
    grid_x = step + grid_x.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[1] - step * 2
    )

    grid_y = grid_y + grid_center[0]
    grid_x = grid_x + grid_center[1]
    xy = torch.stack([grid_x, grid_y], dim=-1).to(device)
    return xy


def sample_pos_embed(grid_size, embed_dim, coords):
    pos_embed = get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size)
    pos_embed = (
        torch.from_numpy(pos_embed)
        .reshape(grid_size[0], grid_size[1], embed_dim)
        .float()
        .unsqueeze(0)
        .to(coords.device)
    )
    sampled_pos_embed = bilinear_sample2d(
        pos_embed.permute(0, 3, 1, 2), coords[:, 0, :, 0], coords[:, 0, :, 1]
    )
    return sampled_pos_embed


class LinearKernel(nn.Module):
    def __init__(self):
        super(LinearKernel, self).__init__()
        self.const = nn.Parameter(torch.ones(1))

    def forward(self, x1, x2, sigma=1e-4):
        numerator = torch.sum(x1.unsqueeze(2) * x2.unsqueeze(1), dim=-1)
        return numerator + self.const


class RBFKernel(nn.Module):
    def __init__(self, input_dim):
        super(RBFKernel, self).__init__()

        self.scale = nn.Parameter(torch.ones(input_dim))

    def forward(self, x1, x2):
        distance = torch.sum((x1.unsqueeze(2) - x2.unsqueeze(1)) ** 2, dim=-1)
        return torch.exp(-distance / (2 * self.scale**2))


class KernelBlock(nn.Module):
    def __init__(self, cfg):
        super(KernelBlock, self).__init__()

        self.kernel_list = cfg.kernel_block.kernel_list
        self.composition = cfg.kernel_block.composition
        assert self.composition in ["sum", "product"]

        kernel_nets = []
        for kernel in self.kernel_list:
            if kernel == "linear":
                kernel_nets.append(LinearKernel())
            elif kernel == "rbf":
                kernel_nets.append(RBFKernel(input_dim=cfg.S))
        self.kernels = nn.ModuleList(kernel_nets)

    def forward(self, features, epsilon=1e-5):
        """
        Compute the kernelized matrix using the specified kernel function.
        :param features: Feature tensor of shape [B, T, C].
        :return: Kernelized matrix of shape [B, T, T].
        """
        if self.composition == "sum":
            K = 0
            for kernel in self.kernels:
                K = K + kernel(features, features)
        elif self.composition == "product":
            K = 1
            for kernel in self.kernels:
                K = K * kernel(features, features)

        K = K + epsilon * torch.eye(K.size(-1)).unsqueeze(0).to(K.device)
        return K


class LeapKernel(nn.Module):
    def __init__(
        self,
        cfg,
        stride=4,
    ):
        super(LeapKernel, self).__init__()
        self.cfg = cfg.model
        self.S = self.cfg.sliding_window_len
        self.stride = stride

        if "anchor_aug" in cfg:
            self.anchor_aug = cfg.anchor_aug
        else:
            self.anchor_aug = None

        self.kernel_from_delta = True
        if "kernel_from_delta" in self.cfg:
            self.kernel_from_delta = self.cfg.kernel_from_delta

        self.interp_shape = (384, 512)
        self.hidden_dim = self.cfg.hidden_dim if "hidden_dim" in self.cfg else 256
        self.latent_dim = self.cfg.latent_dim if "latent_dim" in self.cfg else 128
        self.corr_levels = self.cfg.corr_levels if "corr_levels" in self.cfg else 4
        self.corr_radius = self.cfg.corr_radius if "corr_radius" in self.cfg else 3

        self.add_space_attn = self.cfg.add_space_attn  # default: True
        self.fnet = BasicEncoder(
            output_dim=self.latent_dim, norm_fn="instance", dropout=0, stride=stride
        )

        self.updateformer = UpdateFormer(
            space_depth=self.cfg.space_depth,
            time_depth=self.cfg.time_depth,
            input_dim=456,
            hidden_size=self.cfg.hidden_size,
            num_heads=self.cfg.num_heads,
            output_dim=self.latent_dim + 2,
            mlp_ratio=4.0,
            add_space_attn=self.cfg.add_space_attn,
        )

        self.kernel_block = KernelBlock(cfg=self.cfg)

        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.ffeat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.vis_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
        )

        self.var_predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.Softplus(),
                )
                for _ in range(2)  # for (x,y) dimension
            ]
        )

        # predict dynamic track
        if "motion_label_block" in self.cfg:
            self.motion_label_block = MotionLabelBlock(cfg=self.cfg, S=self.S)
        else:
            self.motion_label_block = None

    def forward_iteration(
        self,
        fmaps,
        coords_init,
        feat_init=None,
        vis_init=None,
        track_mask=None,
        iters=4,
    ):
        B, S_init, N, D = coords_init.shape
        assert D == 2
        assert B == 1

        B, S, __, H8, W8 = fmaps.shape

        device = fmaps.device

        if S_init < S:
            coords = torch.cat(
                [coords_init, coords_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
            vis_init = torch.cat(
                [vis_init, vis_init[:, -1].repeat(1, S - S_init, 1, 1)], dim=1
            )
        else:
            coords = coords_init.clone()

        fcorr_fn = CorrBlock(
            fmaps, num_levels=self.corr_levels, radius=self.corr_radius
        )

        ffeats = feat_init.clone()

        times_ = torch.linspace(0, S - 1, S).reshape(1, S, 1)

        pos_embed = sample_pos_embed(
            grid_size=(H8, W8),
            embed_dim=456,
            coords=coords,
        )
        pos_embed = rearrange(pos_embed, "b e n -> (b n) e").unsqueeze(1)
        times_embed = (
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(456, times_[0]))[None]
            .repeat(B, 1, 1)
            .float()
            .to(device)
        )
        coord_predictions = []
        vars_predictions = []

        for __ in range(iters):
            coords = coords.detach()
            fcorr_fn.corr(ffeats)

            fcorrs = fcorr_fn.sample(coords)  # B, S, N, LRR
            LRR = fcorrs.shape[3]

            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, LRR)
            flows_ = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)

            flows_cat = get_2d_embedding(flows_, 64, cat_coords=True)
            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            if track_mask.shape[1] < vis_init.shape[1]:
                track_mask = torch.cat(
                    [
                        track_mask,
                        torch.zeros_like(track_mask[:, 0]).repeat(
                            1, vis_init.shape[1] - track_mask.shape[1], 1, 1
                        ),
                    ],
                    dim=1,
                )
            concat = (
                torch.cat([track_mask, vis_init], dim=2)
                .permute(0, 2, 1, 3)
                .reshape(B * N, S, 2)
            )

            transformer_input = torch.cat([flows_cat, fcorrs_, ffeats_, concat], dim=2)
            x = transformer_input + pos_embed + times_embed

            x = rearrange(x, "(b n) t d -> b n t d", b=B)

            delta = self.updateformer(x)

            delta = rearrange(delta, " b n t d -> (b n) t d")

            delta_coords_ = delta[:, :, :2]
            delta_feats_ = delta[:, :, 2:]

            delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)
            ffeats_ = ffeats.permute(0, 2, 1, 3).reshape(B * N * S, self.latent_dim)

            ffeats_ = self.ffeat_updater(self.norm(delta_feats_)) + ffeats_

            ffeats = ffeats_.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # B,S,N,C

            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1, 3)
            coord_predictions.append(coords * self.stride)

            # compute kernel
            if self.kernel_from_delta:
                kernel_feat = delta[..., 2:]
            else:
                kernel_feat = rearrange(ffeats, "b s n c -> (b n) s c")

            if self.cfg.kernel_block.add_time:
                times_embed_kernel = (
                    torch.from_numpy(
                        get_1d_sincos_pos_embed_from_grid(
                            kernel_feat.shape[-1], times_[0]
                        )
                    )[None]
                    .repeat(B, 1, 1)
                    .float()
                    .to(device)
                )
                kernel_feat += times_embed_kernel

            kernel_feat_x = self.var_predictors[0](kernel_feat)
            kernel_mat_x = self.kernel_block(kernel_feat_x).reshape(B, N, S, S)
            kernel_feat_y = self.var_predictors[1](kernel_feat)
            kernel_mat_y = self.kernel_block(kernel_feat_y).reshape(B, N, S, S)
            kernel_mat = [kernel_mat_x, kernel_mat_y]
            vars_predictions.append(kernel_mat)

        vis_e = self.vis_predictor(ffeats.reshape(B * S * N, self.latent_dim)).reshape(
            B, S, N
        )

        if self.motion_label_block is not None:
            dynamic_e = self.motion_label_block(ffeats, coords).squeeze(2)
        else:
            dynamic_e = torch.ones(B, N).to(coords.device)

        return coord_predictions, vars_predictions, vis_e, dynamic_e, feat_init

    def forward(self, rgbs, queries, iters=4, feat_init=None, is_train=False):
        B, T, C, H, W = rgbs.shape
        B, N, __ = queries.shape

        device = rgbs.device
        assert B == 1
        # INIT for the first sequence
        # We want to sort points by the first frame they are visible to add them to the tensor of tracked points consequtively
        first_positive_inds = queries[:, :, 0].long()

        __, sort_inds = torch.sort(first_positive_inds[0], dim=0, descending=False)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)
        first_positive_sorted_inds = first_positive_inds[0][sort_inds]

        assert torch.allclose(
            first_positive_inds[0], first_positive_inds[0][sort_inds][inv_sort_inds]
        )

        coords_init = queries[:, :, 1:].reshape(B, 1, N, 2).repeat(
            1, self.S, 1, 1
        ) / float(self.stride)

        rgbs = 2 * (rgbs / 255.0) - 1.0

        traj_e = torch.zeros((B, T, N, 2), device=device)
        vis_e = torch.zeros((B, T, N), device=device)
        cov_x_e = torch.zeros((B, T, N), device=device)
        cov_y_e = torch.zeros((B, T, N), device=device)
        dynamic_e = torch.zeros((B, T, N), device=device)

        ind_array = torch.arange(T, device=device)
        ind_array = ind_array[None, :, None].repeat(B, 1, N)

        track_mask = (ind_array >= first_positive_inds[:, None, :]).unsqueeze(-1)
        vis_init = torch.ones((B, self.S, N, 1), device=device).float() * 10

        ind = 0

        track_mask_ = track_mask[:, :, sort_inds].clone()
        coords_init_ = coords_init[:, :, sort_inds].clone()
        vis_init_ = vis_init[:, :, sort_inds].clone()

        prev_wind_idx = 0
        fmaps_ = None
        vis_predictions = []
        coord_predictions = []
        dynamic_predictions = []
        cov_predictions = []
        wind_inds = []
        while ind < T - self.S // 2:
            rgbs_seq = rgbs[:, ind : ind + self.S]

            S = S_local = rgbs_seq.shape[1]
            if S < self.S:
                rgbs_seq = torch.cat(
                    [rgbs_seq, rgbs_seq[:, -1, None].repeat(1, self.S - S, 1, 1, 1)],
                    dim=1,
                )
                S = rgbs_seq.shape[1]
            rgbs_ = rgbs_seq.reshape(B * S, C, H, W)

            if fmaps_ is None:
                fmaps_ = self.fnet(rgbs_)
            else:
                fmaps_ = torch.cat(
                    [fmaps_[self.S // 2 :], self.fnet(rgbs_[self.S // 2 :])], dim=0
                )
            fmaps = fmaps_.reshape(
                B, S, self.latent_dim, H // self.stride, W // self.stride
            )

            curr_wind_points = torch.nonzero(first_positive_sorted_inds < ind + self.S)
            if curr_wind_points.shape[0] == 0:
                ind = ind + self.S // 2
                continue
            wind_idx = curr_wind_points[-1] + 1

            if wind_idx - prev_wind_idx > 0:
                fmaps_sample = fmaps[
                    :, first_positive_sorted_inds[prev_wind_idx:wind_idx] - ind
                ]

                feat_init_ = bilinear_sample2d(
                    fmaps_sample,
                    coords_init_[:, 0, prev_wind_idx:wind_idx, 0],
                    coords_init_[:, 0, prev_wind_idx:wind_idx, 1],
                ).permute(0, 2, 1)

                feat_init_ = feat_init_.unsqueeze(1).repeat(1, self.S, 1, 1)
                feat_init = smart_cat(feat_init, feat_init_, dim=2)

            if prev_wind_idx > 0:
                new_coords = coords[-1][:, self.S // 2 :] / float(self.stride)

                coords_init_[:, : self.S // 2, :prev_wind_idx] = new_coords
                coords_init_[:, self.S // 2 :, :prev_wind_idx] = new_coords[
                    :, -1
                ].repeat(1, self.S // 2, 1, 1)

                new_vis = vis[:, self.S // 2 :].unsqueeze(-1)
                vis_init_[:, : self.S // 2, :prev_wind_idx] = new_vis
                vis_init_[:, self.S // 2 :, :prev_wind_idx] = new_vis[:, -1].repeat(
                    1, self.S // 2, 1, 1
                )

            if self.anchor_aug is not None and is_train:
                # add anchor during training
                N_anchor = self.anchor_aug.num_anchors
                anchor_queries = get_anchors(rgbs_seq, self.anchor_aug).float()  #

                coords_init_anchor = repeat(
                    anchor_queries[:, :, 1:], "b n c -> b s n c", s=S
                )
                vis_init_anchor = (
                    torch.ones((B, S, N_anchor, 1), device=device).float() * 10
                )

                anchor_frame_id = anchor_queries[..., 0].view(-1).long()
                anchor_fmaps_sample = fmaps[:, anchor_frame_id]
                feat_init_anchor = bilinear_sample2d(
                    anchor_fmaps_sample,
                    anchor_queries[:, :, 1],
                    anchor_queries[:, :, 2],
                ).permute(0, 2, 1)
                feat_init_anchor = feat_init_anchor.unsqueeze(1).repeat(1, self.S, 1, 1)

                anchor_ind_array = torch.arange(S, device=device)
                anchor_ind_array = anchor_ind_array[None, :, None].repeat(
                    B, 1, N_anchor
                )
                anchor_task_mask = (
                    anchor_ind_array >= anchor_frame_id[None, None, :]
                ).unsqueeze(-1)

                coords_init_all = torch.cat(
                    [coords_init_[:, :, :wind_idx], coords_init_anchor], dim=2
                )
                feat_init_all = torch.cat(
                    [feat_init[:, :, :wind_idx], feat_init_anchor], dim=2
                )
                vis_init_all = torch.cat(
                    [vis_init_[:, :, :wind_idx], vis_init_anchor], dim=2
                )
                track_mask_all = torch.cat(
                    [track_mask_[:, ind : ind + self.S, :wind_idx], anchor_task_mask],
                    dim=2,
                )

                coords_all, covs_all, vis_all, dynamic_all, __ = self.forward_iteration(
                    fmaps=fmaps,
                    coords_init=coords_init_all,
                    feat_init=feat_init_all,
                    vis_init=vis_init_all,
                    track_mask=track_mask_all,
                    iters=iters,
                )
                # remove anchors
                coords = [x[:, :, :wind_idx] for x in coords_all]
                covs = [[x[0][:, :wind_idx], x[1][:, :wind_idx]] for x in covs_all]
                vis = vis_all[:, :, :wind_idx]
                dynamic = dynamic_all[:, :wind_idx]

            else:
                coords, covs, vis, dynamic, __ = self.forward_iteration(
                    fmaps=fmaps,
                    coords_init=coords_init_[:, :, :wind_idx],
                    feat_init=feat_init[:, :, :wind_idx],
                    vis_init=vis_init_[:, :, :wind_idx],
                    track_mask=track_mask_[:, ind : ind + self.S, :wind_idx],
                    iters=iters,
                )

            if is_train:
                vis_predictions.append(torch.sigmoid(vis[:, :S_local]))
                dynamic_predictions.append(
                    torch.sigmoid(repeat(dynamic, "b n -> b s n", s=S_local))
                )
                # cov_predictions.append([cov for cov in covs])
                cov_predictions_temp = []
                for cov_mat in covs:
                    cov_predictions_temp.append(
                        [
                            cov_mat[0][:, :, :S_local, :S_local],
                            cov_mat[1][:, :, :S_local, :S_local],
                        ]
                    )
                cov_predictions.append(cov_predictions_temp)
                coord_predictions.append([coord[:, :S_local] for coord in coords])
                wind_inds.append(wind_idx)

            traj_e[:, ind : ind + self.S, :wind_idx] = coords[-1][:, :S_local]
            vis_e[:, ind : ind + self.S, :wind_idx] = vis[:, :S_local]
            cov_x_e[:, ind : ind + self.S, :wind_idx] = torch.diagonal(
                covs[-1][0][:, :, :S_local, :S_local], dim1=-2, dim2=-1
            ).permute(0, 2, 1)
            cov_y_e[:, ind : ind + self.S, :wind_idx] = torch.diagonal(
                covs[-1][1][:, :, :S_local, :S_local], dim1=-2, dim2=-1
            ).permute(0, 2, 1)
            dynamic_e[:, ind : ind + self.S, :wind_idx] = repeat(
                dynamic, "b n -> b s n", s=S_local
            )

            track_mask_[:, : ind + self.S, :wind_idx] = 0.0
            ind = ind + self.S // 2

            prev_wind_idx = wind_idx

        traj_e = traj_e[:, :, inv_sort_inds]
        vis_e = vis_e[:, :, inv_sort_inds]
        cov_x_e = cov_x_e[:, :, inv_sort_inds]
        cov_y_e = cov_y_e[:, :, inv_sort_inds]
        vis_e = torch.sigmoid(vis_e)

        dynamic_e = dynamic_e[:, :, inv_sort_inds]
        dynamic_e = torch.sigmoid(dynamic_e)

        train_data = (
            (
                vis_predictions,
                coord_predictions,
                dynamic_predictions,
                cov_predictions,
                wind_inds,
                sort_inds,
            )
            if is_train
            else None
        )

        return traj_e, feat_init, vis_e, (cov_x_e, cov_y_e), dynamic_e, train_data
