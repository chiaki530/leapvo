import os
import argparse

import numpy as np
import imageio
import torch
import cv2
from tqdm import tqdm

import pdb


def read_image(path):
    """Read image and output RGB image (0-1).
    Args:
        path (str): path to file
    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img



def read_images_from_folder(path):
    image_files = sorted(os.listdir(path))
    frames = []
    for img in image_files:
        frame = read_image(os.path.join(path, img))
        frames.append(frame)
    return np.stack(frames), image_files

def read_video_from_path(path):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)



def video_to_zoedepth(video):
    print("predict monocular depth (ZoeD_N) ...")
    video = video / 255.0   # S, C, H, W
    repo = "/storage/user/chwe/Research/NIPS24/projects/leap3d/thirdparty/ZoeDepth"
    model_zoe_n = torch.hub.load(repo, "ZoeD_N", source="local", pretrained=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)

    depths = []
    for i in tqdm(range(video.shape[0])):
        sample = video[i]
        sample = torch.from_numpy(sample).to(DEVICE).unsqueeze(0).permute(0,3,1,2).float()
        depth_tensor = zoe.infer(sample)
        prediction = depth_tensor.detach().squeeze().cpu().numpy()
        # depths.append(depth_tensor)

        depth_min = prediction.min()
        depth_max = prediction.max()

        # if depth_max - depth_min > np.finfo("float").eps:
        #     out = (prediction - depth_min) / (depth_max - depth_min)
        # else:
        #     out = np.zeros(prediction.shape, dtype=prediction.type)
        depths.append(prediction)

    depths = np.array(depths)
    print(depth_min, depth_max)
    return depths
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", help="path of image folder")
    parser.add_argument("--save_dir", help="path of save folder")
    parser.add_argument("--model", help="depth model")
    args = parser.parse_args()


    if os.path.isdir(args.img_dir):
        video_np, image_files = read_images_from_folder(args.img_dir)
    else:
        video_np = read_video_from_path(args.img_dir)

    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()
    B, S, C, H ,W = video.shape
    video_depth = video_to_zoedepth(video_np).reshape(S, H, W, 1)

    # save depth
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 
    
    assert len(image_files) == S
    for s in range(S):
        save_name = image_files[s].split('.')[0]
        save_path = os.path.join(args.save_dir, save_name)
        np.save(save_path, video_depth[s])
        print("save ZoeDepth to", save_path, video_depth[s].shape)

if __name__ == '__main__':
    main()