import math

import torch
from torchvision.ops import nms
import torch.nn.functional as F

def box_nms(prob, config, remove_zero=None, top_k=None, iou_threshold=0.1):
    ### Batching not supported (only implemented for batch size of 1) ###
    if len(prob.size()) == 3:
        prob = prob[0]
    # Discard detections too close to image border
    box_size_half = round(config["nms_box_size"] / 2.)
    prob[:box_size_half] = 0.
    prob[-box_size_half:] = 0.
    prob[:, :box_size_half] = 0.
    prob[:, -box_size_half:] = 0.

    # Apply NMS on detected keypoints: if remove_zero is set, this threshold will be used instead of detection threshold
    if top_k and not remove_zero:
        remove_zero = 1e-6
    if remove_zero:
        pts_candidates = (prob > remove_zero).nonzero()
        prob = prob[prob > remove_zero].flatten()
    else:
        pts_candidates = (prob > config["detection_threshold"]).nonzero()
        prob = prob[prob > config["detection_threshold"]].flatten()
    boxes_begin = pts_candidates - config["nms_box_size"] / 2.
    boxes_end = pts_candidates + config["nms_box_size"] / 2.
    boxes = torch.cat((boxes_begin, boxes_end), dim=1)
    idxs_filtered = nms(boxes, prob, iou_threshold=iou_threshold)
    if top_k:
        idxs_filtered = idxs_filtered[:top_k]
    return [pts_candidates[idxs_filtered]], [prob[idxs_filtered]]

def fast_nms(prob, config, top_k=None):
    # Find all probablilities that are the largest in their box (does not work for ties, but very unlikely to happen with floats)
    nms_mask = prob == torch.nn.functional.max_pool2d(prob, kernel_size=config["nms_box_size"], stride=1, padding=math.floor(config["nms_box_size"]/2))
    prob[~nms_mask] = 0.

    if top_k:
        det_threshold = 1e-6
    else:
        det_threshold = config["detection_threshold"]
    det_mask = prob > det_threshold
    pts_candidates = det_mask.nonzero()
    prob_candidates = prob[det_mask].flatten()

    pts_batch = [pts_candidates[pts_candidates[:, 0] == i][:, 1:] for i in range(prob.shape[0])]
    prob_batch = [prob_candidates[pts_candidates[:, 0] == i] for i in range(prob.shape[0])]

    if top_k:
        for batch in range(prob.shape[0]):
            prob_batch[batch], sorted_idxs = torch.sort(prob_batch[batch], descending=True)
            pts_batch[batch] = pts_batch[batch][sorted_idxs]
            pts_batch[batch] = pts_batch[batch][:top_k]
            prob_batch[batch] = prob_batch[batch][:top_k]

    return pts_batch, prob_batch

def create_occlusion_mask_FAST(keypoint_tensor: torch.Tensor, 
                               num_to_select: int = 10, 
                               patch_size: int = 5,
                               kernel_size: int = 8, 
                               gt_keypoint_map_batch: torch.Tensor = None) -> torch.Tensor:
        
    if keypoint_tensor.dim() == 3:
        # (B, H, W) -> (B, 1, H, W)
        keypoints_bchw = keypoint_tensor.unsqueeze(1)
        input_was_3d = True
    elif keypoint_tensor.dim() == 4 and keypoint_tensor.shape[1] == 1:
        keypoints_bchw = keypoint_tensor
        input_was_3d = False
    else:
        raise ValueError(f"输入张量形状 {keypoint_tensor.shape} 不支持。"
                         "请输入 (B, H, W) 或 (B, 1, H, W)。")
    
    device = keypoint_tensor.device
    B, C, H, W = keypoints_bchw.shape
    
    if C != 1:
        raise ValueError(f"输入张量的通道 C 必须为 1, 你的为 {C}")
        
    # 1. 二元掩码: 1 = 可见, 0 = 遮挡
    occlusion_mask_bchw = torch.ones_like(keypoints_bchw) 
    # 2. 值掩码: 遮挡区域为随机值, 其他为 0
    value_mask_bchw = torch.zeros_like(keypoints_bchw)

    padding = patch_size // 2

    for b in range(B):
        keypoints_hw = keypoints_bchw[b, 0] 

        if gt_keypoint_map_batch is not None:
            gt_keypoint_map = gt_keypoint_map_batch[b:b+1] #1,1,H,W
            gt_keypoint_map = gt_keypoint_map.clamp(0,1) #only matched gt keypoints
            gt_keypoint_map_unshuffle = F.pixel_unshuffle(gt_keypoint_map, kernel_size)
            gt_keypoint_map_unshuffle = gt_keypoint_map_unshuffle.sum(dim=1).clamp(0,1).unsqueeze(1) #1,1, H/k, W/k
            gt_keypoint_map_unshuffle = gt_keypoint_map_unshuffle.expand(-1,kernel_size*kernel_size,-1,-1)
            gt_keypoint_map_suffle = F.pixel_shuffle(gt_keypoint_map_unshuffle, kernel_size) #matched gt keypoints are allowed
            keypoints_hw = keypoints_hw * gt_keypoint_map_suffle.squeeze()


        coords_hw = torch.nonzero(keypoints_hw) 
        num_keypoints = coords_hw.shape[0]

        if num_keypoints == 0:
            continue 
            
        num_to_select_this_item = min(num_keypoints, num_to_select)
        
        perm = torch.randperm(num_keypoints, device=device)
        selected_indices = perm[:num_to_select_this_item]
        selected_coords = coords_hw[selected_indices] 
        
        for i in range(num_to_select_this_item):
            h = selected_coords[i, 0] # H 索引
            w = selected_coords[i, 1] # W 索引
            
            h_start = torch.clamp(h - padding, 0, H)
            h_end   = torch.clamp(h + padding + 1, 0, H)
            w_start = torch.clamp(w - padding, 0, W)
            w_end   = torch.clamp(w + padding + 1, 0, W)
            
            occlusion_mask_bchw[b, 0, h_start:h_end, w_start:w_end] = 0.0
            # 2. 为这个 patch 生成一个随机值 (0 到 1 之间)
            patch_value = torch.rand(1, device=device) # 这是一个张量, e.g., tensor([0.73])
            
            # 3. 将这个随机值 "涂" 到 value_mask 上
            value_mask_bchw[b, 0, h_start:h_end, w_start:w_end] = patch_value

    if input_was_3d:
        # (B, H, W)
        return occlusion_mask_bchw.squeeze(1), value_mask_bchw.squeeze(1)
    else:
        # (B, 1, H, W)
        return occlusion_mask_bchw, value_mask_bchw
    
def batch_mutual_nearest_neighbor(desc1, desc2):
    """
    PyTorch GPU 版 BFMatcher (双向互匹配)。
    只有 A->B 和 B->A 一致时才保留。
    
    Returns:
        valid_matches: List[Tensor], 长度为 B。每个元素是 (K, 2) 的 Tensor，
                       表示 [idx_in_desc1, idx_in_desc2]
    """
    with torch.no_grad():
        desc1 = F.normalize(desc1, p=2, dim=-1)
        desc2 = F.normalize(desc2, p=2, dim=-1)

        # 1. 计算相似度矩阵 (B, N, M)
        sim_matrix = torch.matmul(desc1, desc2.transpose(-2, -1))

        # 2. 前向匹配 (A -> B): 找到 desc1 对应的最佳 desc2
        # max_dim=2 意味着在 M 维度找最大值
        val_1, idx_1 = torch.max(sim_matrix, dim=2) # shape (B, N)

        # 3. 后向匹配 (B -> A): 找到 desc2 对应的最佳 desc1
        # max_dim=1 意味着在 N 维度找最大值
        val_2, idx_2 = torch.max(sim_matrix, dim=1) # shape (B, M)

        # 4. 互匹配检查 (Cross Check)
        # 我们需要检查: idx_2[idx_1[i]] == i ?
        
        # 这是一个批处理操作技巧：
        # gather 用于从 idx_2 中按 idx_1 的索引取值
        # idx_2_mapped_to_1 的形状是 (B, N)
        idx_2_mapped_to_1 = torch.gather(idx_2, 1, idx_1)
        
        # 创建 range 索引 (B, N) 用于对比
        batch_indices = torch.arange(idx_1.shape[1], device=desc1.device)
        batch_indices = batch_indices.unsqueeze(0).expand(idx_1.shape[0], -1)
        
        # 生成 MNN 掩码 (B, N)
        mutual_mask = (idx_2_mapped_to_1 == batch_indices)

        # 5. 提取匹配对 (格式化输出)
        results = []
        for b in range(sim_matrix.shape[0]):
            # 拿到第 b 个 batch 的掩码
            mask_b = mutual_mask[b]
            
            # desc1 的索引
            indices_a = torch.nonzero(mask_b).squeeze(1)
            # desc2 的索引
            indices_b = idx_1[b][indices_a]
            
            # 堆叠为 (K, 2)
            matches_b = torch.stack([indices_a, indices_b], dim=1)
            results.append(matches_b)
            
        return results

def matches2keypoint_maps(matches:torch.Tensor, img_shape):

    # Create map of zeros with the feature id in its pixel location
    kp_map0 = torch.zeros(img_shape, device=matches[0].device)
    kp_map0[matches[0][:,1].long(), matches[0][:,0].long()] = torch.arange(len(matches[0]),device=matches[0].device).float() + 1

    kp_map1 = torch.zeros(img_shape,device=matches[1].device)
    kp_map1[matches[1][:,1].long(), matches[1][:,0].long()] = torch.arange(len(matches[0]),device=matches[1].device).float() + 1

    return [kp_map0, kp_map1]

def generate_gt_matches(matches0:torch.Tensor, matches1:torch.Tensor):
    """
    Generate ground truth matches from two sets of matches to a common reference.

    Args:
        matches0 (Tensor): (B, N). [x,-1]
        matches1 (Tensor): (B, M). [x,-1]
    Returns:
        gt_matches0 (Tensor): (B, N) [1,-1]
        gt_matches1 (Tensor): (B, M) [1,-1]
        gt_assignment (Tensor): (B,N,M) [0, 1]
    """
    B = matches0.shape[0]
    N = matches0.shape[1]
    M = matches1.shape[1]

    gt_assignment = torch.zeros((B, N, M), device=matches0.device)


    valid = matches0 >= 0  # (B, N, 1)
    gt_assignment[valid] = gt_assignment[valid].scatter_(1, matches0[valid].unsqueeze(1), 1)

    # gt_assignment[common_matches] = 1
    gt_matches0 = matches0.clamp(max=1)
    gt_matches1 = matches1.clamp(max=1)

    return gt_matches0, gt_matches1, gt_assignment

