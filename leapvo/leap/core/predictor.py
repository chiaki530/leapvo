# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from tqdm import tqdm
from .cotracker.cotracker import get_points_on_a_grid, CoTracker
from .model_utils import smart_cat

import pdb


def build_cotracker(
    checkpoint: str,
):
    if checkpoint is None:
        return build_cotracker_stride_4_wind_8()
    model_name = checkpoint.split("/")[-1].split(".")[0]
    if model_name == "cotracker_stride_4_wind_8":
        return build_cotracker_stride_4_wind_8(checkpoint=checkpoint)
    elif model_name == "cotracker_stride_4_wind_12":
        return build_cotracker_stride_4_wind_12(checkpoint=checkpoint)
    elif model_name == "cotracker_stride_8_wind_16":
        return build_cotracker_stride_8_wind_16(checkpoint=checkpoint)
    else:
        raise ValueError(f"Unknown model name {model_name}")


# model used to produce the results in the paper
def build_cotracker_stride_4_wind_8(checkpoint=None):
    return _build_cotracker(
        stride=4,
        sequence_len=8,
        checkpoint=checkpoint,
    )


def build_cotracker_stride_4_wind_12(checkpoint=None):
    return _build_cotracker(
        stride=4,
        sequence_len=12,
        checkpoint=checkpoint,
    )


# the fastest model
def build_cotracker_stride_8_wind_16(checkpoint=None):
    return _build_cotracker(
        stride=8,
        sequence_len=16,
        checkpoint=checkpoint,
    )


def _build_cotracker(
    stride,
    sequence_len,
    checkpoint=None,
):
    cotracker = CoTracker(
        stride=stride,
        S=sequence_len,
        add_space_attn=True,
        space_depth=6,
        time_depth=6,
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        cotracker.load_state_dict(state_dict)
    return cotracker


class CoTrackerPredictor(torch.nn.Module):
    def __init__(
        self, checkpoint="cotracker/checkpoints/cotracker_stride_4_wind_8.pth"
    ):
        super().__init__()
        self.interp_shape = (384, 512)
        self.support_grid_size = 6
        model = build_cotracker(checkpoint)

        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        video,  # (1, T, 3, H, W)
        # input prompt types:
        # - None. Dense tracks are computed in this case. You can adjust *query_frame* to compute tracks starting from a specific frame.
        # *backward_tracking=True* will compute tracks in both directions.
        # - queries. Queried points of shape (1, N, 3) in format (t, x, y) for frame index and pixel coordinates.
        # - grid_size. Grid of N*N points from the first frame. if segm_mask is provided, then computed only for the mask.
        # You can adjust *query_frame* and *backward_tracking* for the regular grid in the same way as for dense tracks.
        queries: torch.Tensor = None,
        segm_mask: torch.Tensor = None,  # Segmentation mask of shape (B, 1, H, W)
        grid_size: int = 0,
        grid_query_frame: int = 0,  # only for dense and regular grid tracks
        backward_tracking: bool = False,
    ):

        if queries is None and grid_size == 0:
            tracks, visibilities = self._compute_dense_tracks(
                video,
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
            )
        else:
            tracks, visibilities = self._compute_sparse_tracks(
                video,
                queries,
                segm_mask,
                grid_size,
                add_support_grid=(grid_size == 0 or segm_mask is not None),
                grid_query_frame=grid_query_frame,
                backward_tracking=backward_tracking,
            )

        return tracks, visibilities

    def _compute_dense_tracks(
        self, video, grid_query_frame, grid_size=30, backward_tracking=False
    ):
        *_, H, W = video.shape
        grid_step = W // grid_size
        grid_width = W // grid_step
        grid_height = H // grid_step
        tracks = visibilities = None
        grid_pts = torch.zeros((1, grid_width * grid_height, 3)).to(video.device)
        grid_pts[0, :, 0] = grid_query_frame
        for offset in tqdm(range(grid_step * grid_step)):
            ox = offset % grid_step
            oy = offset // grid_step
            grid_pts[0, :, 1] = (
                torch.arange(grid_width).repeat(grid_height) * grid_step + ox
            )
            grid_pts[0, :, 2] = (
                torch.arange(grid_height).repeat_interleave(grid_width) * grid_step + oy
            )
            tracks_step, visibilities_step = self._compute_sparse_tracks(
                video=video,
                queries=grid_pts,
                backward_tracking=backward_tracking,
            )
            tracks = smart_cat(tracks, tracks_step, dim=2)
            visibilities = smart_cat(visibilities, visibilities_step, dim=2)

        return tracks, visibilities

    def _compute_sparse_tracks(
        self,
        video,
        queries,
        segm_mask=None,
        grid_size=0,
        add_support_grid=False,
        grid_query_frame=0,
        backward_tracking=False,
    ):
        B, T, C, H, W = video.shape
        assert B == 1

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(video, tuple(self.interp_shape), mode="bilinear")     # self.interp_shape = (384, 512)
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])

        if queries is not None:
            queries = queries.clone()
            B, N, D = queries.shape
            assert D == 3
            # scale query position according to interp_shape
            queries[:, :, 1] *= self.interp_shape[1] / W
            queries[:, :, 2] *= self.interp_shape[0] / H
        elif grid_size > 0:
            grid_pts = get_points_on_a_grid(grid_size, self.interp_shape, device=video.device)
            if segm_mask is not None:
                segm_mask = F.interpolate(
                    segm_mask, tuple(self.interp_shape), mode="nearest"
                )
                point_mask = segm_mask[0, 0][
                    (grid_pts[0, :, 1]).round().long().cpu(),
                    (grid_pts[0, :, 0]).round().long().cpu(),
                ].bool()
                grid_pts = grid_pts[:, point_mask]

            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                dim=2,
            )

        if add_support_grid:
            grid_pts = get_points_on_a_grid(self.support_grid_size, self.interp_shape, device=video.device)
            grid_pts = torch.cat(
                [torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2
            )
            queries = torch.cat([queries, grid_pts], dim=1)

        tracks, __, visibilities, __ = self.model(rgbs=video, queries=queries, iters=6)

        if backward_tracking:
            tracks, visibilities = self._compute_backward_tracks(
                video, queries, tracks, visibilities
            )
            if add_support_grid:
                queries[:, -self.support_grid_size ** 2 :, 0] = T - 1
        if add_support_grid:
            tracks = tracks[:, :, : -self.support_grid_size ** 2]
            visibilities = visibilities[:, :, : -self.support_grid_size ** 2]
        thr = 0.9
        visibilities = visibilities > thr

        # correct query-point predictions
        # see https://github.com/facebookresearch/co-tracker/issues/28

        # TODO: batchify
        for i in range(len(queries)):
            queries_t = queries[i, :tracks.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))

            # overwrite the predictions with the query points
            tracks[i, queries_t, arange] = queries[i, :tracks.size(2), 1:]

            # correct visibilities, the query points should be visible
            visibilities[i, queries_t, arange] = True

        tracks[:, :, :, 0] *= W / float(self.interp_shape[1])
        tracks[:, :, :, 1] *= H / float(self.interp_shape[0])
        return tracks, visibilities

    def _compute_backward_tracks(self, video, queries, tracks, visibilities):
        inv_video = video.flip(1).clone()
        inv_queries = queries.clone()
        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

        inv_tracks, __, inv_visibilities, __ = self.model(
            rgbs=inv_video, queries=inv_queries, iters=6
        )

        inv_tracks = inv_tracks.flip(1)
        inv_visibilities = inv_visibilities.flip(1)

        mask = tracks == 0

        tracks[mask] = inv_tracks[mask]
        visibilities[mask[:, :, :, 0]] = inv_visibilities[mask[:, :, :, 0]]
        return tracks, visibilities
