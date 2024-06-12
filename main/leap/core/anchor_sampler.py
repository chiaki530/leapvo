import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def meshgrid2d(B, Y, X, stack=False, norm=False, device="cuda"):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y - 1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


def image_gradient(images):
    # gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
    images_pad = F.pad(images, (1, 1, 1, 1), "constant", 0)
    gray = images_pad.sum(dim=2)
    dx = gray[..., :-1, 1:] - gray[..., :-1, :-1]
    dy = gray[..., 1:, :-1] - gray[..., :-1, :-1]
    g = torch.sqrt(dx**2 + dy**2)
    g = F.avg_pool2d(g, 4, 4)
    return g


def get_anchors(rgbs, cfg):
    """

    Args:
        rgbs (_type_): [B, S, C, H, W]

    Returns:
        xys: [B, N, 2]
        xys: [B, N]
    """
    B, S, C, ht, wd = rgbs.shape
    anchor_mode = cfg.anchor_mode
    num_anchors = cfg.num_anchors
    margin = cfg.margin if "margin" in cfg else 64
    device = rgbs.device

    frame = 0
    if "frame" in cfg:
        frame = cfg.frame

    if anchor_mode in ["random", "uniform"]:
        if anchor_mode == "random":
            x = torch.randint(margin, wd - margin, size=[B, num_anchors])
            y = torch.randint(margin, ht - margin, size=[B, num_anchors])
            xys = torch.stack([x, y], dim=-1).float().to(device)
        elif anchor_mode == "uniform":
            M_ = np.sqrt(num_anchors).round().astype(np.int32)
            grid_y, grid_x = meshgrid2d(
                1, M_, M_, stack=False, norm=False, device=device
            )
            grid_y = margin + grid_y.reshape(1, -1) / float(M_ - 1) * (ht - 2 * margin)
            grid_x = margin + grid_x.reshape(1, -1) / float(M_ - 1) * (wd - 2 * margin)
            xys = torch.stack([grid_x, grid_y], dim=-1).to(device)  # B, N_*N_, 2
            xys = repeat(xys, "i n c -> (b i) n c", b=B)
        if frame >= 0:
            xys_sid = torch.ones((B, num_anchors), device=device) * frame
        else:
            xys_sid = torch.randint(0, S, size=[B, num_anchors])

    elif anchor_mode == "orb":
        xys = []
        for b in range(B):
            assert frame >= 0
            # pytorch to
            image_array = rgbs[b, frame].permute(1, 2, 0).detach().cpu().numpy()
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            gray = ((gray - gray.min()) * (255.0 / (gray.max() - gray.min()))).astype(
                np.uint8
            )
            orb = cv2.ORB_create()
            kps = orb.detect(gray, None)
            kps, des = orb.compute(gray, kps)
            kp_np = []

            # Iterate over each keypoint
            kps_pt = [x.pt for x in kps]
            kp_np = np.array(kps_pt)
            if len(kps_pt) > 0:
                mask = (
                    (kp_np[..., 0] > margin)
                    & (kp_np[..., 0] < wd - margin)
                    & (kp_np[..., 1] > margin)
                    & (kp_np[..., 1] < ht - margin)
                )
                kp_np = kp_np[mask]
                np.random.shuffle(kp_np)
                kp_np = kp_np[:num_anchors]

            if len(kp_np) == num_anchors:
                kp_np = np.array(kp_np)
                xys.append(torch.from_numpy(kp_np).to(device))
            else:
                # if detector is smaller than num_anchors, add random points
                diff = num_anchors - len(kp_np)
                x = torch.randint(margin, wd - margin, size=[diff])
                y = torch.randint(margin, ht - margin, size=[diff])
                xy = torch.stack([x, y], dim=-1).float().to(device)
                kp_np = np.array(kp_np)
                kp_np = torch.concat([xy, torch.from_numpy(kp_np).to(device)], dim=0)
                xys.append(kp_np)
        xys = torch.stack(xys, dim=0).to(device).float()
        xys_sid = torch.ones((B, num_anchors), device=device) * frame

    elif anchor_mode == "sift":
        xys = []
        for b in range(B):
            assert frame >= 0
            image_array = rgbs[b, frame].permute(1, 2, 0).detach().cpu().numpy()
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            gray = ((gray - gray.min()) * (255.0 / (gray.max() - gray.min()))).astype(
                np.uint8
            )
            sift = cv2.SIFT_create()

            kps = sift.detect(gray, None)
            kps, des = sift.compute(gray, kps)

            kp_np = []
            # Iterate over each keypoint
            kps_pt = [x.pt for x in kps]
            kp_np = np.array(kps_pt)
            if len(kps_pt) > 0:
                mask = (
                    (kp_np[..., 0] > margin)
                    & (kp_np[..., 0] < wd - margin)
                    & (kp_np[..., 1] > margin)
                    & (kp_np[..., 1] < ht - margin)
                )
                kp_np = kp_np[mask]
                np.random.shuffle(kp_np)
                kp_np = kp_np[:num_anchors]

            # Convert the list to a numpy array
            # img2 = cv2.drawKeypoints(gray, kps, None, color=(0,255,0), flags=0)
            # plt.imshow(img2)
            # plt.scatter(kp_np[...,0], kp_np[...,1], color='skyblue')
            # plt.savefig('debug/pytorch_harris/debug.png')
            # plt.close()

            if len(kp_np) == num_anchors:
                xys.append(torch.from_numpy(kp_np).to(device))
            else:
                # if detector is smaller than num_anchors, add random points
                diff = num_anchors - len(kp_np)
                x = torch.randint(margin, wd - margin, size=[diff])
                y = torch.randint(margin, ht - margin, size=[diff])
                xy = torch.stack([x, y], dim=-1).float().to(device)
                kp_np = torch.concat([xy, torch.from_numpy(kp_np).to(device)], dim=0)
                xys.append(kp_np)

        xys = torch.stack(xys, dim=0).to(device).float()
        xys_sid = torch.ones((B, num_anchors), device=device) * frame

    elif anchor_mode == "img_grad":
        g = image_gradient(rgbs)
        g = g.view(B * S, 1, ht // 4, wd // 4)
        x = torch.randint(0, wd - 1, size=[B * S, 4 * num_anchors], device=device)
        y = torch.randint(1, ht - 1, size=[B * S, 4 * num_anchors], device=device)
        xx = x / (wd - 1) * 2.0 - 1.0
        yy = y / (ht - 1) * 2.0 - 1.0

        coords = torch.stack([xx, yy], dim=-1).float()
        g = F.grid_sample(
            g, coords.view(B * S, 1, -1, 2), mode="bilinear", align_corners=True
        )
        g = g[:, 0, 0, :]
        ix = torch.argsort(g, dim=1)
        x = torch.gather(x, 1, ix[:, -num_anchors:])
        y = torch.gather(y, 1, ix[:, -num_anchors:])
        xys = torch.stack([x, y], dim=-1).float()
        xys = xys.view(B, S, -1, 2)
        xys = xys[:, frame]
        xys_sid = torch.ones((B, num_anchors), device=device) * frame

    elif "grid_grad" in anchor_mode:
        g = image_gradient(rgbs)
        g = g.view(B * S, 1, ht // 4, wd // 4)

        grid_size = int(anchor_mode.split("_")[-1])
        H_grid, W_grid = ht // grid_size, wd // grid_size

        num_anchors_grid = num_anchors // (grid_size * grid_size)
        g_patch = rearrange(
            g, "b c (g1 h) (g2 w) -> (b g1 g2) c h w", g1=grid_size, g2=grid_size
        )
        xx = torch.randint(
            0,
            W_grid,
            size=[B * S * grid_size * grid_size, 8 * num_anchors_grid],
            device=device,
        )
        yy = torch.randint(
            0,
            H_grid,
            size=[B * S * grid_size * grid_size, 8 * num_anchors_grid],
            device=device,
        )
        x = xx / (W_grid - 1) * 2.0 - 1.0
        y = yy / (H_grid - 1) * 2.0 - 1.0
        coords = torch.stack([x, y], dim=-1).float()
        coords = coords.view(B * S * grid_size * grid_size, 1, -1, 2)
        gg_patch = F.grid_sample(g_patch, coords, mode="bilinear", align_corners=True)
        gg_patch = gg_patch[:, 0, 0]
        ix_patch = torch.argsort(gg_patch, dim=1)
        x = torch.gather(xx, 1, ix_patch[:, -num_anchors_grid:])
        y = torch.gather(yy, 1, ix_patch[:, -num_anchors_grid:])

        offset = torch.linspace(0, grid_size - 1, grid_size)
        offset_y, offset_x = torch.meshgrid(offset, offset)
        offset = torch.stack([offset_x, offset_y], dim=-1).to(device)
        offset = offset.view(-1, 2)
        offset[..., 0] = offset[..., 0] * W_grid
        offset[..., 1] = offset[..., 1] * H_grid
        x_global = x.view(B, S, -1, num_anchors_grid) + offset[..., 0].view(1, 1, -1, 1)
        y_global = y.view(B, S, -1, num_anchors_grid) + offset[..., 1].view(1, 1, -1, 1)
        xys = torch.stack([x_global, y_global], dim=-1).float()
        xys = xys.view(B, S, -1, 2)
        xys = xys[:, frame]
        xys_sid = torch.ones((B, num_anchors), device=device) * frame

    elif "max_grad" in anchor_mode:
        g = image_gradient(rgbs)
        g = g.view(B * S, 1, ht // 4, wd // 4)

        grid_size = int(anchor_mode.split("_")[-1])
        H_grid, W_grid = ht // grid_size, wd // grid_size
        num_anchors_grid = num_anchors // (grid_size * grid_size)
        g_patch = rearrange(
            g, "b c (g1 h) (g2 w) -> (b g1 g2) c h w", g1=grid_size, g2=grid_size
        )
        gg_patch = rearrange(g_patch, "b c h w -> (b c) (h w)")
        ix_patch = torch.argsort(gg_patch, dim=1)[:, -num_anchors_grid:]
        iw_patch_x = 4 * (ix_patch % (wd // (4 * grid_size)))
        iw_patch_y = 4 * (ix_patch // (wd // (4 * grid_size)))

        offset = torch.linspace(0, grid_size - 1, grid_size)
        offset_y, offset_x = torch.meshgrid(offset, offset)
        offset = torch.stack([offset_x, offset_y], dim=-1).to(device)
        offset = offset.view(-1, 2)
        offset[..., 0] = offset[..., 0] * W_grid
        offset[..., 1] = offset[..., 1] * H_grid

        x_global = iw_patch_x.view(B, S, -1, num_anchors_grid) + offset[..., 0].view(
            1, 1, -1, 1
        )
        y_global = iw_patch_y.view(B, S, -1, num_anchors_grid) + offset[..., 1].view(
            1, 1, -1, 1
        )

        xys = torch.stack([x_global, y_global], dim=-1).float()
        xys = xys.view(B, S, -1, 2)
        xys = xys[:, frame]
        xys_sid = torch.ones((B, num_anchors), device=device) * frame

    queries = torch.cat([xys_sid.unsqueeze(-1), xys], dim=2)
    return queries
