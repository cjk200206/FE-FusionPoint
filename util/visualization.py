import cv2
import math
import numpy as np
import torch

def ts2image(ts):
    #ts = np.average(ts, axis=2)  # Take average over channels
    #ts = np.uint8(255. * ts)
    #ts = cv2.cvtColor(ts, cv2.COLOR_GRAY2BGR)
    ts_out = np.ones(list(ts.shape[:2]) + [3])
    # blue_values = ts[ts[..., 2] > 0][..., 2].T
    # red_values = ts[ts[..., 7] > 0][..., 7].T
    # ts_out[ts[..., 2] > 0] = np.array([np.ones_like(blue_values), 1. - blue_values, 1. - blue_values]).T
    # ts_out[ts[..., 7] > 0] = np.array([1. - red_values, 1. - red_values, np.ones_like(red_values)]).T
    
    if ts.shape[2] == 10:
        i = 3; j = 8
        # i = 1; j = 6

    elif ts.shape[2] == 6:
        i = 2; j = 5
    
    blue_values = ts[ts[..., i] > 0][..., i].T
    red_values = ts[ts[..., j] > 0][..., j].T
    ts_out[ts[..., i] > 0] = np.array([np.ones_like(blue_values), 1. - blue_values, 1. - blue_values]).T
    ts_out[ts[..., j] > 0] = np.array([1. - red_values, 1. - red_values, np.ones_like(red_values)]).T
    ts_out = np.rint(ts_out * 255.).astype(np.uint8)
    return ts_out

def kp_map2matches(kp_maps):
    match_list = []
    for kp_map in kp_maps:
        num_matches = round(np.max(kp_map))
        matches = [np.where(kp_map == i + 1) for i in range(num_matches)]
        match_list.append(matches)
    min_len = min([len(matches) for matches in match_list])
    match_list = [matches[:min_len] for matches in match_list]
    valid_match_mask = np.ones(min_len, dtype=bool)
    for i in range(min_len):
        for matches in match_list:
            if not len(matches[i][0]) or not len(matches[i][1]):
                valid_match_mask[i] = False
    for i in reversed(range(min_len)):
        if not valid_match_mask[i]:
            for matches in match_list:
                del matches[i]
    return match_list

def visualize_time_surface(ts):
    ts = ts2image(ts)

    cv2.imshow("Time Surface", ts)
    cv2.waitKey(0)

def visualize_matches(kp_maps, img0, img1, title="Matches", waitkey=True, return_image=False, resize_factor=1.0):
    matches = kp_map2matches(kp_maps)
    kp0 = [cv2.KeyPoint(float(resize_factor * p[1]), float(resize_factor * p[0]), 1) for p in matches[0]]
    kp1 = [cv2.KeyPoint(float(resize_factor * p[1]), float(resize_factor * p[0]), 1) for p in matches[1]]
    cv_matches = [cv2.DMatch(i, i, 0.0) for i in range(len(kp0))]

    # Add unmatched keypoints
    if np.sum(kp_maps[0] < 0) > 1:
        kp0 = kp0 + [cv2.KeyPoint(float(resize_factor * p[1]), float(resize_factor * p[0]), 1) for p in np.nonzero(kp_maps[0] < 0)]
    if np.sum(kp_maps[1] < 0) > 1:
        kp1 = kp1 + [cv2.KeyPoint(float(resize_factor * p[1]), float(resize_factor * p[0]), 1) for p in np.nonzero(kp_maps[1] < 0)]
    
    for kp in kp0:
        img0 = cv2.circle(img0, (round(kp.pt[0]), round(kp.pt[1])), radius=math.ceil(3*resize_factor), color=(17, 86, 187), thickness=math.ceil(resize_factor))
    for kp in kp1:
        img1 = cv2.circle(img1, (round(kp.pt[0]), round(kp.pt[1])), radius=math.ceil(3*resize_factor), color=(17, 86, 187), thickness=math.ceil(resize_factor))
    matched_img = cv2.drawMatches(img0, kp0, img1, kp1, cv_matches,
                                    None, matchColor=(17, 86, 187),
                                    singlePointColor=(17, 86, 187),
                                    matchesThickness=round(resize_factor),
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if return_image:
        frames_det = np.hstack([img0, img1])
        return matched_img, frames_det
    else:
        cv2.imshow(title, matched_img)
        if waitkey:
            cv2.waitKey(0)
        return None, None

def resize_and_make_border(img, resize_factor=1.0, border_size=1):
    if border_size == 0:
        return

    # Resize
    border_size *= math.ceil(resize_factor)
    img = cv2.resize(img, (round(resize_factor * img.shape[1]), round(resize_factor * img.shape[0])),
                                interpolation=cv2.INTER_NEAREST)
    
    # Border
    img = img[border_size:-border_size, border_size:-border_size]
    img = cv2.copyMakeBorder(
        img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return img

def visualize_and_overlay_heatmap(original_img, feature_map, alpha=0.5, colormap=cv2.COLORMAP_JET, mean = False):
    """
    将特征图转换为热力图并叠加到原图上。

    Args:
        original_img (np.ndarray): 原始图像，形状为 (H, W, C)，BGR格式，uint8类型。
        feature_map (torch.Tensor): 网络的中间层输出，形状为 (C, H, W)。
        alpha (float): 热力图的透明度。
        colormap (int): OpenCV 的色彩映射方案。
    """
    # 1. 降维 & CPU & NumPy
    if mean:
        heatmap = feature_map.mean(dim=0) # 在通道维度上取平均
    else:
        heatmap = torch.norm(feature_map, p=2, dim=0)
    heatmap = heatmap.cpu().numpy()

    # 2. 归一化到 [0, 1]
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    
    # 3. 缩放到 [0, 255] 并转换为 uint8
    heatmap_uint8 = np.uint8(255 * heatmap)

    # 4. 应用色彩映射
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # 5. 调整热力图尺寸以匹配原图
    height, width, _ = original_img.shape
    heatmap_resized = cv2.resize(heatmap_color, (width, height))

    # 6. 叠加热力图和原图
    overlaid_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_resized, alpha, 0)

    return overlaid_img, heatmap_resized

# def visualize_and_overlay_heatmap(original_img, feature_map, alpha=0.5, colormap=cv2.COLORMAP_JET, mean=False):
#     """
#     Args:
#         feature_map (torch.Tensor): (C, H, W)
#     """
#     # 1. 通道降维策略优化
#     # 对于中间层特征，max 往往比 mean 更能反映“最显著特征”
#     # 如果有很多死寂通道(0值)，mean会被拉得很低，max则不会
#     if mean:
#         heatmap = feature_map.mean(dim=0)
#     else:
#         # 建议尝试用 max 替代 norm，除非你明确需要 L2 能量
#         heatmap, _ = torch.max(feature_map, dim=0) 
#         # 或者保持你原来的: heatmap = torch.norm(feature_map, p=2, dim=0)

#     heatmap = heatmap.detach().cpu().numpy()

#     # ================= [核心修改] 鲁棒归一化 =================
#     # 计算 1% 和 99% 分位数，而不是 min/max
#     # 这能忽略掉那 1% 的极端亮点（比如右上角那个）
#     lower = np.percentile(heatmap, 1)
#     upper = np.percentile(heatmap, 99)
    
#     # 截断（Clip）数值到这个区间
#     heatmap = np.clip(heatmap, lower, upper)
    
#     # 在截断后的范围内归一化 (加 eps 防止除以 0)
#     heatmap = (heatmap - lower) / (upper - lower + 1e-8)
#     # =======================================================

#     # 3. 后续步骤保持不变
#     heatmap_uint8 = np.uint8(255 * heatmap)
#     heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

#     height, width, _ = original_img.shape
#     heatmap_resized = cv2.resize(heatmap_color, (width, height))
#     overlaid_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_resized, alpha, 0)

#     return overlaid_img, heatmap_resized