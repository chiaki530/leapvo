import os

import cv2
import matplotlib as mpl
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from moviepy.editor import ImageSequenceClip


def read_video_from_path(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
    else:
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                break
        cap.release()
    return np.stack(frames)


class SLAMVisualizer:
    def __init__(
        self,
        cfg,
        save_dir=None,
        # grayscale: bool = False,
        # pad_value: int = 0,
        # fps: int = 10,
        # mode: str = "rainbow",  # 'cool', 'optical_flow'
        # linewidth: int = 2,
        # show_first_frame: int = 10,
        # tracks_leave_trace: int = 0,  # -1 for infinite
    ):
        self.cfg = cfg.visualizer
        self.cfg_full = cfg
        self.mode = self.cfg.mode
        self.save_dir = self.cfg.save_dir
        if save_dir is not None:
            self.save_dir = save_dir

        if self.cfg.mode == "rainbow":
            self.color_map = mpl.colormaps["gist_rainbow"]
        elif self.cfg.mode == "cool":
            self.color_map = mpl.colormaps[self.cfg.mode]
        self.show_first_frame = self.cfg.show_first_frame
        self.grayscale = self.cfg.grayscale
        self.tracks_leave_trace = self.cfg.tracks_leave_trace
        self.pad_value = self.cfg.pad_value
        self.linewidth = self.cfg.linewidth
        self.fps = self.cfg.fps

        # storage

        self.frames = []
        self.tracks = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def add_track(self, track):
        self.tracks.append(track)

    def draw_tracks_on_frames(self):
        video = torch.stack(self.frames, dim=0)
        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )
        video = video.permute(0, 2, 3, 1).detach().cpu().numpy()

        res_video_sta = []
        res_video_dyn = []
        # process input video
        for rgb in video:
            res_video_sta.append(rgb.copy())
            res_video_dyn.append(rgb.copy())

        T = self.fps * 2  # period of color repetition

        for t, track in enumerate(self.tracks):
            targets = track["targets"][0].long().detach().cpu().numpy()
            targets = targets + self.pad_value
            S, N, _ = targets.shape

            vis_label = None
            static_label = None
            coords_vars = None

            if "vis_label" in track:
                vis_label = track["vis_label"][0].detach().cpu().numpy()
            if "static_label" in track:
                static_label = track["static_label"][0].detach().cpu().numpy()
            if "coords_vars" in track:
                coords_vars = track["coords_vars"][0].detach().cpu().numpy()

            for s in range(S):
                color = (
                    np.array(self.color_map(((t - S + 1 + s) % T) / T)[:3])[None] * 255
                )
                vector_colors = np.repeat(color, N, axis=0)

                for n in range(N):
                    coord = (targets[s, n, 0], targets[s, n, 1])
                    visibile = True
                    if vis_label is not None:
                        visibile = vis_label[s, n]
                    static = True
                    if static_label is not None:
                        static = static_label[s, n]
                    if coords_vars is not None:
                        # conf_scale = np.sqrt(coords_vars[s,n]) * 3
                        conf_scale = 4 - 3 * np.exp(-coords_vars[s, n])
                    else:
                        conf_scale = 1.0

                    if coord[0] != 0 and coord[1] != 0:

                        radius = int(self.linewidth * 2)
                        if static:
                            cv2.circle(
                                res_video_sta[t],
                                coord,
                                radius,
                                vector_colors[n].tolist(),
                                thickness=-1 if visibile else 2 - 1,
                            )
                            cv2.circle(
                                res_video_sta[t],
                                coord,
                                int(radius * conf_scale * 3),
                                vector_colors[n].tolist(),
                                2 - 1,
                            )
                        else:
                            cv2.circle(
                                res_video_dyn[t],
                                coord,
                                radius,
                                vector_colors[n].tolist(),
                                thickness=-1 if visibile else 2 - 1,
                            )
        #  construct the final rgb sequence

        res_video = []
        for i in range(len(video)):
            frame_combine = np.concatenate([res_video_sta[i], res_video_dyn[i]], axis=0)
            res_video.append(frame_combine)

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def save_video(self, filename, writer=None, step=0):
        video = self.draw_tracks_on_frames()

        # export video
        if writer is not None:
            writer.add_video(
                f"{filename}_pred_track",
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
            clip = ImageSequenceClip(wide_list[2:-1], fps=self.fps)

            # Write the video file
            save_path = os.path.join(self.save_dir, f"{filename}_pred_track.mp4")
            clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)

            print(f"Video saved to {save_path}")


class LEAPVisualizer(SLAMVisualizer):
    def __init__(
        self,
        cfg,
        save_dir=None,
    ):
        super(LEAPVisualizer, self).__init__(cfg=cfg, save_dir=save_dir)

    def add_frame(self, frame):
        self.frames.append(frame)

    def add_track(self, track):
        self.tracks.append(track)

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        alpha: float = 0.5,
    ):
        T, N, _ = tracks.shape

        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].tolist(),
                        self.linewidth,
                        cv2.LINE_AA,
                    )
            if self.tracks_leave_trace > 0:
                rgb = cv2.addWeighted(rgb, alpha, original, 1 - alpha, 0)
        return rgb

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        vector_colors: torch.Tensor = None,
        visibility: torch.Tensor = None,
        variances: torch.Tensor = None,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2

        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())

        # vector_colors = np.zeros((T, N, 3))

        # for t in range(T):
        #     color = np.array(self.color_map(t / T)[:3])[None] * 255
        #     vector_colors[t] = np.repeat(color, N, axis=0)

        #  draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(1, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]

                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                )

        #  draw points
        for t in range(T):
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]

                if coord[0] != 0 and coord[1] != 0:
                    cv2.circle(
                        res_video[t],
                        coord,
                        int(self.linewidth * 2),
                        vector_colors[t, i].tolist(),
                        thickness=-1 if visibile else 2 - 1,
                    )

                    # draw uncertainty
                    if variances is not None:
                        # conf_scale = np.sqrt(coords_vars[s,n]) * 3
                        conf_scale = 4 - 3 * np.exp(-variances[0, t, i])
                        overlay = res_video[t].copy()
                        cv2.circle(
                            overlay,
                            coord,
                            int(self.linewidth * 2 * conf_scale * 3),
                            vector_colors[t, i].tolist(),
                            1,
                            -1,
                        )
                        alpha = 0.5
                        res_video[t] = cv2.addWeighted(
                            overlay, alpha, res_video[t], 1 - alpha, 0
                        )

        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)

    def draw_tracks_on_frames(self):
        video = torch.stack(self.frames, dim=0)
        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )

        res_video_sta = video.clone()
        res_video_dyn = video.clone()

        T = self.fps  # period of color repetition

        for t, track in enumerate(self.tracks):
            fid = track["fid"]
            targets = track["targets"] + self.pad_value
            weights = track["weights"]
            queries = track["queries"]
            vis_label = track["vis_label"]
            B, S, S1, M, C = targets.shape

            vector_colors = np.zeros((S, S1, M, 3))
            for s1 in range(S1):
                kf_stride = self.cfg_full.slam.kf_stride
                fid_norm = ((fid // kf_stride + s1) % T) / T
                color = np.array(self.color_map(fid_norm)[:3]) * 255
                vector_colors[:, s1] = repeat(color, "c -> s m c", s=S, m=M)
                # vector_colors[t] = repeat()

            if "coords_vars" in track:
                variances = track["coords_vars"]

            # plot uncertanity
            variances = track["coords_vars"]
            var_mean = variances.mean(dim=1)
            high_var_th = torch.quantile(var_mean, 0.9)
            high_mask = var_mean[0] > high_var_th
            variances = variances / variances.mean()  # normalized

            dyn_rgbs = res_video_dyn[fid - S : fid][None]
            dyn_tracks = targets.reshape(B, S, -1, C)[:, :, high_mask]
            dyn_vis_label = vis_label[:, :, high_mask]
            dyn_colors = vector_colors.reshape(S, -1, 3)[
                :, high_mask.detach().cpu().numpy()
            ]

            dyn_color = mpl.colors.to_rgba("yellow")
            dyn_colors[..., 0] = dyn_color[0] * 255
            dyn_colors[..., 1] = dyn_color[1] * 255
            dyn_colors[..., 2] = dyn_color[2] * 255

            dyn_var = (
                variances[:, :, high_mask].detach().cpu().numpy()
                if variances is not None
                else None
            )

            res_video = self.draw_tracks_on_video(
                video=dyn_rgbs,
                tracks=dyn_tracks,
                visibility=dyn_vis_label,
                vector_colors=dyn_colors,
                variances=None,  # dyn_var
            )
            res_video_dyn[fid - S : fid] = res_video

            variances = None

            if "static_label" in track:
                static_label = track["static_label"]

                dyn_rgbs = res_video_dyn[fid - S : fid][None]
                # check dynamic mask of the full track
                static_mask = static_label[0].float().mean(dim=0) < 0.5

                dyn_tracks = targets.reshape(B, S, -1, C)[:, :, static_mask]
                dyn_vis_label = vis_label[:, :, static_mask]
                dyn_colors = vector_colors.reshape(S, -1, 3)[
                    :, static_mask.detach().cpu().numpy()
                ]

                dyn_color = mpl.colors.to_rgba("red")
                dyn_colors[..., 0] = dyn_color[0] * 255
                dyn_colors[..., 1] = dyn_color[1] * 255
                dyn_colors[..., 2] = dyn_color[2] * 255

                dyn_var = (
                    variances[:, :, static_mask].detach().cpu().numpy()
                    if variances is not None
                    else None
                )

                res_video = self.draw_tracks_on_video(
                    video=dyn_rgbs,
                    tracks=dyn_tracks,
                    visibility=dyn_vis_label,
                    vector_colors=dyn_colors,
                    variances=dyn_var,
                )
                res_video_dyn[fid - S : fid] = res_video

                rgbs = res_video_sta[fid - S : fid][None]
                sta_tracks = targets.reshape(B, S, -1, C)[:, :, ~static_mask]
                sta_vis_label = vis_label[:, :, ~static_mask]
                sta_colors = vector_colors.reshape(S, -1, 3)[
                    :, ~static_mask.detach().cpu().numpy()
                ]

                # use one color
                sta_color = mpl.colors.to_rgba("lawngreen")
                sta_colors[..., 0] = sta_color[0] * 255
                sta_colors[..., 1] = sta_color[1] * 255
                sta_colors[..., 2] = sta_color[2] * 255

                sta_var = (
                    variances[:, :, ~static_mask].detach().cpu().numpy()
                    if variances is not None
                    else None
                )
                res_video = self.draw_tracks_on_video(
                    video=rgbs,
                    tracks=sta_tracks,
                    visibility=sta_vis_label,
                    vector_colors=sta_colors,
                    variances=sta_var,
                )
                res_video_sta[fid - S : fid] = res_video
            else:
                rgbs = res_video_sta[fid - S : fid][None]
                res_video = self.draw_tracks_on_video(
                    video=rgbs,
                    tracks=targets.reshape(B, S, -1, C),
                    visibility=vis_label,
                    vector_colors=vector_colors.reshape(S, -1, 3),
                )

                res_video_sta[fid - S : fid] = res_video

        res_video = torch.cat([res_video_sta, res_video_dyn], dim=-2)

        return res_video[None].byte()

    def save_video(self, filename, writer=None, step=0):
        video = self.draw_tracks_on_frames()

        # export video
        if writer is not None:
            writer.add_video(
                f"{filename}_pred_track",
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
            clip = ImageSequenceClip(wide_list[2:-1], fps=self.fps)

            # Write the video file
            save_path = os.path.join(self.save_dir, f"{filename}_pred_track.mp4")
            clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)

            print(f"Video saved to {save_path}")
