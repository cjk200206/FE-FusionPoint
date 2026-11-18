import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch

# from models.util import box_nms
# from util.train_utils import list2device

def fix_seed():
    import random
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

def extract_keypoints_and_descriptors(pts, desc_map, detection_threshold, resize_factor=1.0):
    if len(pts) > 0 and not len(pts[0]) == 2:
        # Keypoint map provided
        pts = np.where(pts > detection_threshold)
        pts = np.array(pts).transpose()
    descriptors = desc_map[pts[:, 0], pts[:, 1]]
    kpts = [cv2.KeyPoint(float(resize_factor * pts[i][1]), float(resize_factor * pts[i][0]), 1) for i in range(len(pts))]
    return kpts, descriptors

def extract_keypoints_and_descriptors_torch(pts, desc_map, resize_factor=1.0):
    """   
    Args:
        pts: (N, 2) Tensor [y, x] ,
        desc_map: (C, H, W) Tensor,
        resize_factor: 
        
    Returns:
        kpts: (N, 2) Tensor [x, y],
        descriptors: (N, C) Tensor
    """
    if pts.shape[0] == 0:
        device = desc_map.device
        C = desc_map.shape[0]
        return torch.empty((0, 2), device=device), torch.empty((0, C), device=device)
    # pts [y, x]
    pts_long = pts.long()
    descriptors = desc_map[:, pts_long[:, 0], pts_long[:, 1]].t()

    y = pts[:, 0].float()
    x = pts[:, 1].float()
    #(N, 2) -> [x, y]
    kpts = torch.stack([x, y], dim=1) * resize_factor

    return kpts, descriptors