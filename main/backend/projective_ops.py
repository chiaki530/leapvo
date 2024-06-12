import torch
import torch.nn.functional as F

from .lietorch import SE3, Sim3

MIN_DEPTH = 0.2


def extract_intrinsics(intrinsics):
    return intrinsics[..., None, None, :].unbind(dim=-1)


def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(), torch.arange(wd).to(**kwargs).float()
    )

    return torch.stack([x, y], dim=-1)


def iproj(patches, intrinsics):
    """inverse projection"""
    x, y, d = patches.unbind(dim=2)
    fx, fy, cx, cy = intrinsics[..., None, None].unbind(dim=2)

    i = torch.ones_like(d)
    xn = (x - cx) / fx
    yn = (y - cy) / fy

    X = torch.stack([xn, yn, i, d], dim=-1)
    return X


def proj(X, intrinsics, depth=False):
    """projection"""

    X, Y, Z, W = X.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[..., None, None].unbind(dim=2)

    # d = 0.01 * torch.ones_like(Z)
    # d[Z > 0.01] = 1.0 / Z[Z > 0.01]
    # d = torch.ones_like(Z)
    # d[Z.abs() > 0.1] = 1.0 / Z[Z.abs() > 0.1]

    d = 1.0 / Z.clamp(min=0.1)
    x = fx * (d * X) + cx
    y = fy * (d * Y) + cy

    if depth:
        return torch.stack([x, y, d], dim=-1)

    return torch.stack([x, y], dim=-1)


def transform(
    poses,
    patches,
    intrinsics,
    ii,
    jj,
    kk,
    depth=False,
    valid=False,
    jacobian=False,
    tonly=False,
):
    """projective transform"""

    # backproject
    X0 = iproj(patches[:, kk], intrinsics[:, ii])

    # transform
    Gij = poses[:, jj] * poses[:, ii].inv()

    if tonly:
        Gij[..., 3:] = torch.as_tensor([0, 0, 0, 1], device=Gij.device)

    X1 = Gij[:, :, None, None] * X0

    # project
    x1 = proj(X1, intrinsics[:, jj], depth)

    if jacobian:
        p = X1.shape[2]
        X, Y, Z, H = X1[..., p // 2, p // 2, :].unbind(dim=-1)
        o = torch.zeros_like(H)
        i = torch.zeros_like(H)

        fx, fy, cx, cy = intrinsics[:, jj].unbind(dim=-1)

        d = torch.zeros_like(Z)
        d[Z.abs() > 0.2] = 1.0 / Z[Z.abs() > 0.2]

        Ja = torch.stack(
            [
                H,
                o,
                o,
                o,
                Z,
                -Y,
                o,
                H,
                o,
                -Z,
                o,
                X,
                o,
                o,
                H,
                Y,
                -X,
                o,
                o,
                o,
                o,
                o,
                o,
                o,
            ],
            dim=-1,
        ).view(1, len(ii), 4, 6)

        Jp = torch.stack(
            [
                fx * d,
                o,
                -fx * X * d * d,
                o,
                o,
                fy * d,
                -fy * Y * d * d,
                o,
            ],
            dim=-1,
        ).view(1, len(ii), 2, 4)

        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:, :, None].adjT(Jj)

        Jz = torch.matmul(Jp, Gij.matrix()[..., :, 3:])

        return x1, (Z > 0.2).float(), (Ji, Jj, Jz)

    if valid:
        return x1, (X1[..., 2] > 0.2).float()

    return x1


def point_cloud(poses, patches, intrinsics, ix):
    """generate point cloud from patches"""
    return poses[:, ix, None, None].inv() * iproj(patches, intrinsics[:, ix])


def flow_mag(poses, patches, intrinsics, ii, jj, kk, beta=0.3):
    """projective transform"""

    coords0 = transform(poses, patches, intrinsics, ii, ii, kk)
    coords1 = transform(poses, patches, intrinsics, ii, jj, kk, tonly=False)
    coords2 = transform(poses, patches, intrinsics, ii, jj, kk, tonly=True)

    flow1 = (coords1 - coords0).norm(dim=-1)
    flow2 = (coords2 - coords0).norm(dim=-1)

    return beta * flow1 + (1 - beta) * flow2


#######################################################################################3
# my utils
from einops import rearrange, repeat


def back_proj(xy, xy_depth, intrinsics, cams_c2w=None):
    """_summary_

    Args:
        xy (_type_): B, N, 2
        xy_depth (_type_): B, N, 1
        intrinsics (_type_): B, 4
        cams_c2w (_type_): B, 4, 4
    Returns:
        P: B, N, 4
    """
    fx, fy, cx, cy = (
        intrinsics[:, 0],
        intrinsics[:, 1],
        intrinsics[:, 2],
        intrinsics[:, 3],
    )
    X = (xy[..., 0] - cx) / fx
    Y = (xy[..., 1] - cy) / fy
    i = torch.ones_like(X)
    D = xy_depth[..., 0]
    P = torch.stack([X * D, Y * D, D, i], dim=2)
    if cams_c2w is not None:
        P = cams_c2w.float() @ P.permute(0, 2, 1)
        P = P.permute(0, 2, 1)
    return P


def proj_to_frames(P, intrinsics, cams_w2c):
    """project from world to cameras

    Args:
        P (_type_): B, N, 4
        intrinsics (_type_): B, S, 4
        cams_w2c (_type_): B, S, 4, 4

    Returns:
        _type_: _description_
    """
    B, N = P.shape[:2]
    S = cams_w2c.shape[1]
    fx, fy, cx, cy = (
        intrinsics[..., [0]],
        intrinsics[..., [1]],
        intrinsics[..., [2]],
        intrinsics[..., [3]],
    )
    P = repeat(P, "b n c -> b s n c", s=S)
    P_c = cams_w2c.float() @ P.permute(0, 1, 3, 2)
    P_c = P_c.permute(0, 1, 3, 2)
    X_c = P_c[..., 0]
    Y_c = P_c[..., 1]
    D_c = P_c[..., 2]
    d_c = 1.0 / D_c
    x_c = fx * (X_c * d_c) + cx
    y_c = fy * (Y_c * d_c) + cy
    xy_c = torch.stack([x_c, y_c], dim=-1)
    return xy_c
