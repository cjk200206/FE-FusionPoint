import random

import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from models.util import batch_mutual_nearest_neighbor, matches2keypoint_maps

def double_softmax_distance(desc_0, desc_1, temperature=1.0):
    # SiLK (GPLv3), Copyright (c) Meta Platforms, Inc. and affiliates.
    # https://github.com/facebookresearch/silk/blob/main/lib/matching/mnn.py#L36
    # See LICENSE and THIRD_PARTY_LICENSES.txt.
    similarity = torch.matmul(desc_0, desc_1.T) / temperature
    matching_probability = torch.softmax(similarity, dim=0) * torch.softmax(
        similarity, dim=1
    )
    return 1.0 - matching_probability

def detector_loss(logits, ground_truth_keypoint_map, config, mask = None):
    # Make keypoints with label -1 positive so they can be used for training
    ground_truth_keypoint_map = torch.abs(ground_truth_keypoint_map)

    if config["pixel_wise_predictions"]:
        labels = ground_truth_keypoint_map.clamp(max=1.)  # Make binary
        loss = sigmoid_focal_loss(logits, labels, alpha=config["detector_loss_alpha"],
                                  gamma=config["detector_loss_gamma"], reduction="mean")
        return loss

    else:  # original SuperPoint loss
        # Convert the boolean keypoint map to indices + "no interest point" dustbin
        labels = ground_truth_keypoint_map[:, None, :, :]
        labels = torch.nn.functional.pixel_unshuffle(labels, config["grid_size"])
        shape = list(labels.shape)
        shape[1] = 1  # Set channel size to 1
        labels = torch.cat((2*labels, torch.ones(shape, device=labels.device)), dim=1)  # add "no interest point" dustbin
        # Add a small random matrix to break ties in argmax
        labels = torch.argmax(labels + 0.1 * torch.rand(labels.shape, device=labels.device), dim=1)

        loss = torch.nn.CrossEntropyLoss(reduction='none')
        cross_entropy = loss(logits, labels)
        
        if mask is not None:
            mask = F.max_pool2d(mask.float(), kernel_size=config["grid_size"], stride=config["grid_size"])[:,0,:,:]
            cross_entropy = (cross_entropy * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            cross_entropy = cross_entropy.mean()

        return cross_entropy
    
def detector_loss_event_mask(logits, ground_truth_keypoint_map, config, event_mask = None):
    #make matched to 2
    ground_truth_keypoint_map = ground_truth_keypoint_map.clamp(max=2)

    # Make keypoints with label -1 positive so they can be used for training
    ground_truth_keypoint_map = torch.abs(ground_truth_keypoint_map)

    # original SuperPoint loss
    # Convert the boolean keypoint map to indices + "no interest point" dustbin
    labels_2d = ground_truth_keypoint_map[:, None, :, :]

    #new ver
    if event_mask is not None:
        labels_2d = labels_2d + event_mask.float()

    labels = torch.nn.functional.pixel_unshuffle(labels_2d, config["grid_size"])
    shape = list(labels.shape)
    shape[1] = 1  # Set channel size to 1
    labels = torch.cat((2*labels, torch.ones(shape, device=labels.device)), dim=1)  # add "no interest point" dustbin
    # Add a small random matrix to break ties in argmax
    labels = torch.argmax(labels + 0.1 * torch.rand(labels.shape, device=labels.device), dim=1)

    loss = torch.nn.CrossEntropyLoss(reduction='none')
    cross_entropy = loss(logits, labels)
    

    # new ver
    if event_mask is not None:
        label_mask = F.max_pool2d(labels_2d.float(), kernel_size=config["grid_size"], stride=config["grid_size"])[:,0,:,:]
        label_mask_pos = (label_mask>=2).float() #matched keypoints + f&e points + both

        ##keep only mask & label_point + label_dustbin
        mask_point = label_mask_pos * ((labels != 64).float())
        mask_dustbin = (~label_mask_pos.bool())*(labels == 64).float()
        mask = mask_point + mask_dustbin

        cross_entropy = (cross_entropy * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        cross_entropy = cross_entropy.mean()

    # old ver
    # if event_mask is not None:
    #     event_mask = F.max_pool2d(event_mask.float(), kernel_size=config["grid_size"], stride=config["grid_size"])[:,0,:,:]

    #     ##keep only mask_e & label_point + label_dustbin
    #     mask_point = event_mask * ((labels != 64).float())
    #     mask_dustbin = (~event_mask.bool())*(labels == 64).float()
    #     mask = mask_point + mask_dustbin

    #     cross_entropy = (cross_entropy * mask).sum() / mask.sum().clamp(min=1.0)
    # else:
    #     cross_entropy = cross_entropy.mean()

    return cross_entropy

# def descriptor_loss(descriptors0, descriptors1, keypoint_map0, keypoint_map1, config, mask0=None, mask1=None):
#     # Corresponding keypoints should have an descriptor distance of 0 while all other combinations
#     # should have a descriptor distance of 1
#     # Grid cells that do not contain a desciptor should be not used for loss calculation
    
#     # Reduce keypoint map resolution to descriptor map resolution
#     batch_size, _, Hc, Wc, = descriptors0.shape
#     if config["pixel_wise_predictions"]:
#         Hc = Hc // config["grid_size"]
#         Wc = Wc // config["grid_size"]

#     # Filter out unmatched keypoints (label is -1) and  add channel: shape is [B, C, H, W]
#     keypoint_map0 = keypoint_map0.clamp(min=0.)[:, None, :, :]
#     keypoint_map1 = keypoint_map1.clamp(min=0.)[:, None, :, :]

#     # Get feature with highest ID if there are multiple in the same cell: shape is [B, C, Hc, Wc]
#     keypoint_map_grid0, kp_indeces0 = torch.nn.functional.max_pool2d(keypoint_map0, config["grid_size"], stride=config["grid_size"], return_indices=True)
#     keypoint_map_grid1, kp_indeces1 = torch.nn.functional.max_pool2d(keypoint_map1, config["grid_size"], stride=config["grid_size"], return_indices=True)

#     # Select the descriptors at the keypoints
#     if config["pixel_wise_predictions"]:
#         descriptors0 = descriptors0.flatten(start_dim=2)[:, :, kp_indeces0.flatten()].reshape((batch_size, -1, Hc, Wc))
#         descriptors1 = descriptors1.flatten(start_dim=2)[:, :, kp_indeces1.flatten()].reshape((batch_size, -1, Hc, Wc))

#     # Get difference between two grids. 0 difference means same index (no features will be filtered out
#     # by vaid mask)
#     keypoint_map_grid0 = torch.reshape(keypoint_map_grid0, (batch_size, Hc, Wc, 1, 1))
#     keypoint_map_grid1 = torch.reshape(keypoint_map_grid1, (batch_size, 1, 1, Hc, Wc))
#     kp_correspondence = keypoint_map_grid0 - keypoint_map_grid1
#     s = torch.isclose(kp_correspondence, torch.tensor([0.], device=kp_correspondence.device)).float()
#     # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) corresponds to
#     # point (h', w') and 0 otherwise

#     # Create keypoint map to filter out invalid descriptors
#     keypoint_grid_mask0 = keypoint_map_grid0.clamp(max=1.)
#     keypoint_grid_mask1 = keypoint_map_grid1.clamp(max=1.)
    
#     if mask0 is not None:
#         mask0 = F.max_pool2d(mask0.float(), kernel_size=config["grid_size"], stride=config["grid_size"])[:,0,:,:]
#         keypoint_grid_mask0 = keypoint_grid_mask0 * mask0.unsqueeze(-1).unsqueeze(-1)
#         mask1 = F.max_pool2d(mask1.float(), kernel_size=config["grid_size"], stride=config["grid_size"])[:,0,:,:]
#         keypoint_grid_mask1 = keypoint_grid_mask1 * mask1.unsqueeze(1).unsqueeze(1)

#     merged_keypoint_grid_mask = keypoint_grid_mask0 * keypoint_grid_mask1

#     # Normalize the descriptors and compute their pairwise dot product
#     descriptors0 = descriptors0[:, :, :, :, None, None]
#     descriptors1 = descriptors1[:, :, None, None, :, :]
#     descriptors0 = torch.nn.functional.normalize(descriptors0, dim=1)
#     descriptors1 = torch.nn.functional.normalize(descriptors1, dim=1)
#     dot_product_desc = torch.sum(descriptors0 * descriptors1, dim=1)
#     dot_product_desc = torch.nn.functional.relu(dot_product_desc)
#     # Normalize over image grids
#     dot_product_desc = torch.reshape(
#         torch.nn.functional.normalize(
#             torch.reshape(dot_product_desc, (batch_size, Hc, Wc, Hc*Wc))
#             , dim=3)
#         , (batch_size, Hc, Wc, Hc, Wc))
#     dot_product_desc = torch.reshape(
#         torch.nn.functional.normalize(
#             torch.reshape(dot_product_desc, (batch_size, Hc*Wc, Hc, Wc))
#             , dim=1)
#         , (batch_size, Hc, Wc, Hc, Wc))
#     # dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
#     # descriptor at position (h, w) in the original descriptors map and the
#     # descriptor at position (h', w') in the second image

#     # Compute the loss (filter out descriptors without keypoints in pseudo gt)
#     positive_dist = (config["positive_margin"] - dot_product_desc).clamp(min=0.) * merged_keypoint_grid_mask
#     if config.get("keep_unmatched_descriptors_in_loss", False):
#         # 1. 仅保留左图必须是关键点的约束 (我们需要优化左图的关键点特征)
#         # keypoint_grid_mask0: [B, H, W, 1, 1]
#         valid_source_mask = keypoint_grid_mask0 
        
#         # 2. 计算 Negative Loss (让关键点与右图的所有位置都不相似，除了匹配点)
#         # s 是匹配矩阵 (0/1)
#         # 负样本损失：在非匹配位置 (1-s)，且源点是关键点的地方计算
#         # 去掉了对 mask1 的依赖，这意味着右图是背景的地方也要算 Loss！
#         negative_dist = (dot_product_desc - config["negative_margin"]).clamp(min=0.) * valid_source_mask
        
#     else:
#         negative_dist = (dot_product_desc - config["negative_margin"]).clamp(min=0.) * merged_keypoint_grid_mask
    
#     loss = config["lambda_d"] * s * positive_dist + (1 - s) * negative_dist
#     normalization = (torch.sum(keypoint_grid_mask0)).clamp(min=1.)
#     loss = torch.sum(loss / normalization)

#     # Only for tuning and debugging
#     with torch.no_grad():
#         pos_loss = torch.sum(config["lambda_d"] * s * positive_dist ) / normalization
#         neg_loss = torch.sum((1 - s) * negative_dist ) / normalization
    
#     return loss, pos_loss, neg_loss


def descriptor_loss(descriptors0, descriptors1, keypoint_map0, keypoint_map1, config, mask0=None, mask1=None):
    # ... [前半部分预处理代码完全保持不变] ...
    # Reduce keypoint map resolution to descriptor map resolution
    batch_size, _, Hc, Wc, = descriptors0.shape
    if config["pixel_wise_predictions"]:
        Hc = Hc // config["grid_size"]
        Wc = Wc // config["grid_size"]

    # Filter out unmatched keypoints (label is -1) and  add channel: shape is [B, C, H, W]
    keypoint_map0 = keypoint_map0.clamp(min=0.)[:, None, :, :]
    keypoint_map1 = keypoint_map1.clamp(min=0.)[:, None, :, :]

    # Get feature with highest ID if there are multiple in the same cell: shape is [B, C, Hc, Wc]
    keypoint_map_grid0, kp_indeces0 = torch.nn.functional.max_pool2d(keypoint_map0, config["grid_size"], stride=config["grid_size"], return_indices=True)
    keypoint_map_grid1, kp_indeces1 = torch.nn.functional.max_pool2d(keypoint_map1, config["grid_size"], stride=config["grid_size"], return_indices=True)

    # Select the descriptors at the keypoints
    if config["pixel_wise_predictions"]:
        descriptors0 = descriptors0.flatten(start_dim=2)[:, :, kp_indeces0.flatten()].reshape((batch_size, -1, Hc, Wc))
        descriptors1 = descriptors1.flatten(start_dim=2)[:, :, kp_indeces1.flatten()].reshape((batch_size, -1, Hc, Wc))

    # Get difference between two grids. 0 difference means same index
    keypoint_map_grid0 = torch.reshape(keypoint_map_grid0, (batch_size, Hc, Wc, 1, 1))
    keypoint_map_grid1 = torch.reshape(keypoint_map_grid1, (batch_size, 1, 1, Hc, Wc))
    kp_correspondence = keypoint_map_grid0 - keypoint_map_grid1
    s = torch.isclose(kp_correspondence, torch.tensor([0.], device=kp_correspondence.device)).float()

    # Create keypoint map to filter out invalid descriptors
    keypoint_grid_mask0 = keypoint_map_grid0.clamp(max=1.)
    keypoint_grid_mask1 = keypoint_map_grid1.clamp(max=1.)
    
    if mask0 is not None:
        mask0 = F.max_pool2d(mask0.float(), kernel_size=config["grid_size"], stride=config["grid_size"])[:,0,:,:]
        keypoint_grid_mask0 = keypoint_grid_mask0 * mask0.unsqueeze(-1).unsqueeze(-1)
        mask1 = F.max_pool2d(mask1.float(), kernel_size=config["grid_size"], stride=config["grid_size"])[:,0,:,:]
        keypoint_grid_mask1 = keypoint_grid_mask1 * mask1.unsqueeze(1).unsqueeze(1)

    merged_keypoint_grid_mask = keypoint_grid_mask0 * keypoint_grid_mask1

    # Normalize the descriptors and compute their pairwise dot product
    descriptors0 = descriptors0[:, :, :, :, None, None]
    descriptors1 = descriptors1[:, :, None, None, :, :]
    descriptors0 = torch.nn.functional.normalize(descriptors0, dim=1)
    descriptors1 = torch.nn.functional.normalize(descriptors1, dim=1)
    dot_product_desc = torch.sum(descriptors0 * descriptors1, dim=1)
    dot_product_desc = torch.nn.functional.relu(dot_product_desc)
    
    # Normalize over image grids
    dot_product_desc = torch.reshape(
        torch.nn.functional.normalize(
            torch.reshape(dot_product_desc, (batch_size, Hc, Wc, Hc*Wc))
            , dim=3)
        , (batch_size, Hc, Wc, Hc, Wc))
    dot_product_desc = torch.reshape(
        torch.nn.functional.normalize(
            torch.reshape(dot_product_desc, (batch_size, Hc*Wc, Hc, Wc))
            , dim=1)
        , (batch_size, Hc, Wc, Hc, Wc))

    # ==========================================================
    # Compute the loss (这里开始改动)
    # ==========================================================
    
    # 1. 计算 Positive Distance (正样本部分，完全不动)
    positive_dist = (config["positive_margin"] - dot_product_desc).clamp(min=0.) * merged_keypoint_grid_mask
    
    # 2. 归一化分母 (提前计算，用于后面共用)
    normalization = (torch.sum(keypoint_grid_mask0)).clamp(min=1.)

    # ================== 新增分支：Hard Negative Mining ==================
    if config.get("use_hard_negative_mining", False):
        # [逻辑说明]
        # s=1 代表匹配点。为了找最难负样本，我们要找 "非匹配点(s=0)中分数最高的"。
        # 技巧：将匹配点(s=1)的分数减去一个大数(2.0)，这样取 max 时它们绝对不会被选中。
        
        # dot_product: [B, H, W, H, W]
        # s: [B, H, W, H, W]
        mining_scores = dot_product_desc - (2.0 * s)
        
        # 展平后两维(右图的空间维度)，然后在维度3上取最大值
        # 结果 shape: [B, H, W] -> 代表左图每个点对应的“最难负样本分数”
        hardest_neg_scores, _ = torch.max(mining_scores.reshape(batch_size, Hc, Wc, -1), dim=3)
        
        # 计算负样本损失
        # 这里只约束 anchor (左图) 必须是有效关键点。
        # squeeze: keypoint_grid_mask0 原来是 [B, H, W, 1, 1], 需要变成 [B, H, W]
        valid_anchor_mask = keypoint_grid_mask0.view(batch_size, Hc, Wc)
        negative_dist = (hardest_neg_scores - config["negative_margin"]).clamp(min=0.) * valid_anchor_mask

        # 组合 Loss
        # 注意：pos 是 5D tensor, neg 是 3D tensor，为了避免维度广播错误，这里分别求和
        sum_pos_loss = torch.sum(config["lambda_d"] * s * positive_dist)
        sum_neg_loss = torch.sum(negative_dist)
        
        loss = (sum_pos_loss + sum_neg_loss) / normalization

        # Debug logging variables
        with torch.no_grad():
            pos_loss = sum_pos_loss / normalization
            neg_loss = sum_neg_loss / normalization

    # ================== 原始分支：Global Average Negative ==================
    else:
        # [原始逻辑，保持原样]
        if config.get("keep_unmatched_descriptors_in_loss", False):
            # keypoint_grid_mask0: [B, H, W, 1, 1]
            valid_source_mask = keypoint_grid_mask0 
            negative_dist = (dot_product_desc - config["negative_margin"]).clamp(min=0.) * valid_source_mask
        else:
            negative_dist = (dot_product_desc - config["negative_margin"]).clamp(min=0.) * merged_keypoint_grid_mask
        
        loss = config["lambda_d"] * s * positive_dist + (1 - s) * negative_dist
        loss = torch.sum(loss / normalization)

        # Debug logging variables
        with torch.no_grad():
            pos_loss = torch.sum(config["lambda_d"] * s * positive_dist ) / normalization
            neg_loss = torch.sum((1 - s) * negative_dist ) / normalization
    
    return loss, pos_loss, neg_loss


def dense_descriptor_loss(descriptors0, descriptors1, pixel_mapping, config, mask0=None, mask1=None):
    """
    Args:
        descriptors0: [B, C, Hc, Wc] - 变换图特征 (Warped/Source) -> 对应 mask0
        descriptors1: [B, C, Hc, Wc] - 原图特征 (Original/Target)   -> 对应 mask1 (通常全1)
        pixel_mapping: [B, 2, H_full, W_full] - ts0 对应的原始坐标
        config: 配置字典
        mask0: [B, 1, H_full, W_full] - ts0 的有效区域 mask
    """
    
    # 1. 基础维度
    batch_size, channel, Hc, Wc = descriptors0.shape
    device = descriptors0.device
    num_cells = Hc * Wc
    
    # 2. 坐标对齐 (Coordinates)
    
    # [A] 下采样 Pixel Mapping (Warped Image 每个 Grid 对应的物理坐标)
    # [B, 2, H, W] -> [B, 2, Hc, Wc]
    grid0 = F.interpolate(pixel_mapping, size=(Hc, Wc), mode='bilinear', align_corners=False)
    
    # [B] 生成 Original Image 的标准中心坐标
    uy, ux = torch.meshgrid(torch.arange(Hc, device=device), torch.arange(Wc, device=device), indexing='ij')
    grid1_y = uy.float() * config["grid_size"] + config["grid_size"] // 2
    grid1_x = ux.float() * config["grid_size"] + config["grid_size"] // 2
    # [B, 2, Hc, Wc]
    grid1 = torch.stack((grid1_x, grid1_y), dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # 3. 展平 (Flatten)
    # [B, 2, Hc, Wc] -> [B, N, 2]
    coord0_flat = grid0.flatten(2).transpose(1, 2) 
    coord1_flat = grid1.flatten(2).transpose(1, 2)

    # 4. 计算距离矩阵 (Pairwise Distance)
    # [B, N, N]
    dist_matrix = torch.cdist(coord0_flat, coord1_flat, p=2)

    # 5. 生成匹配标签 (Correspondence Matrix S)
    # 参考 TF 代码: less_equal(dist, grid_size - 0.5)
    dist_threshold = config["grid_size"] - 0.5
    s = (dist_matrix <= dist_threshold).float()

    # 6. 处理 Mask (重要修改)
    # mask0 对应 descriptors0 (Warped)，这是我们需要主要过滤的
    if mask0 is not None:
        # 下采样 mask: [B, 1, Hc, Wc] -> [B, N, 1]
        mask0 = F.max_pool2d(mask0.float(), kernel_size=config["grid_size"], stride=config["grid_size"])
        mask0_flat = mask0.flatten(2).transpose(1, 2)
    else:
        mask0_flat = torch.ones(batch_size, num_cells, 1, device=device)
    
    # TF 代码逻辑中 valid_mask 广播到了整个 loss 矩阵
    # mask0_flat [B, N, 1] 会在计算时广播成 [B, N, N]，即屏蔽掉 Warped 图中无效的行
    valid_mask = mask0_flat 

    # 7. 描述子相似度 (Dot Product)
    desc0_flat = F.normalize(descriptors0.flatten(2), dim=1) # [B, C, N]
    desc1_flat = F.normalize(descriptors1.flatten(2), dim=1) # [B, C, N]

    # [B, N, N]
    dot_product_desc = torch.bmm(desc0_flat.transpose(1, 2), desc1_flat)
    dot_product_desc = F.relu(dot_product_desc)

    # 8. 计算 Loss Matrix (Hinge Loss)
    positive_dist = (config["positive_margin"] - dot_product_desc).clamp(min=0.)
    negative_dist = (dot_product_desc - config["negative_margin"]).clamp(min=0.)
    
    # [B, N, N]
    loss_matrix = config["lambda_d"] * s * positive_dist + (1 - s) * negative_dist

    # 9. 归一化 (Normalization) - 核心修改点
    # 参考 TF: normalization = tf.reduce_sum(valid_mask) * tf.to_float(Hc * Wc)
    
    # Part A: 统计 Warped 图中有多少个有效的 Cell (跨 Batch 求和)
    num_valid_warped_cells = torch.sum(valid_mask) 
    
    # Part B: Original 图的总 Cell 数 (常数 N)
    num_original_cells = float(num_cells)
    
    # 总分母：有效对的总数 (Valid Warped Cells * All Original Cells)
    normalization = num_valid_warped_cells * num_original_cells
    
    # 防止除零
    normalization = normalization.clamp(min=1.0)

    # 10. 最终 Loss
    # mask 只作用于 loss 矩阵的行 (Warped Image 的维度)
    # sum(loss * mask) / normalization
    loss = torch.sum(loss_matrix * valid_mask) / normalization

    # Debug info
    with torch.no_grad():
        pos_term = torch.sum((config["lambda_d"] * s * positive_dist) * valid_mask) / normalization
        neg_term = torch.sum(((1 - s) * negative_dist) * valid_mask) / normalization

    return loss, pos_term, neg_term

def double_softmax_loss(descriptors0, descriptors1, keypoint_map0, keypoint_map1, config):
    max_num_keypoints = 500  # probably never reached, just to prevent OOM
    with torch.no_grad():
        batch_size = descriptors0.shape[0]

        # Count matches for each sample
        num_matches_per_sample0 = torch.abs(keypoint_map0.clamp(min=-1, max=1)).sum(dim=(1,2), dtype=int).clamp(max=max_num_keypoints)
        num_matches_per_sample1 = torch.abs(keypoint_map1.clamp(min=-1, max=1)).sum(dim=(1,2), dtype=int).clamp(max=max_num_keypoints)
        cum_matches_until_sample0 = torch.cumsum(torch.cat((torch.tensor([0], device=num_matches_per_sample0.device), num_matches_per_sample0)), dim=0)
        cum_matches_until_sample1 = torch.cumsum(torch.cat((torch.tensor([0], device=num_matches_per_sample1.device), num_matches_per_sample1)), dim=0)

        # Sanity check
        if torch.max(num_matches_per_sample0) > max_num_keypoints or torch.max(num_matches_per_sample1) > max_num_keypoints:
            print(f"Not all keypoints will be used for evaluation: {torch.max(num_matches_per_sample0)} or {torch.max(num_matches_per_sample1)} exceeds threshold of {max_num_keypoints} keypoints.")
                                             
        # Get indexes of keypoint labels
        kp_idxs0 = torch.nonzero(keypoint_map0, as_tuple=True)
        kp_idxs1 = torch.nonzero(keypoint_map1, as_tuple=True)

        # Only select pixels with keypoints in pseudo gt
        match_indeces0 = keypoint_map0[kp_idxs0]
        match_indeces1 = keypoint_map1[kp_idxs1]

        # Set negative match indeces (= no match in pseudo gt) to nan so they are not matched
        match_indeces0[match_indeces0<0] = torch.nan
        match_indeces1[match_indeces1<1] = torch.nan

        # Create tensor with constant shape
        batched_match_idxs0 = torch.nan * torch.ones(batch_size, max_num_keypoints, device=match_indeces0.device)
        batched_match_idxs1 = torch.nan * torch.ones(batch_size, max_num_keypoints, device=match_indeces1.device)
        for batch in range(batch_size):
            batched_match_idxs0[batch, :num_matches_per_sample0[batch]] = match_indeces0[cum_matches_until_sample0[batch]:cum_matches_until_sample0[batch+1]]
            batched_match_idxs1[batch, :num_matches_per_sample1[batch]] = match_indeces1[cum_matches_until_sample1[batch]:cum_matches_until_sample1[batch+1]]

        # Get difference between two grids. 0 difference means same index (no features will be filtered out
        # by vaid mask)
        batched_match_idxs0 = torch.reshape(batched_match_idxs0, (batch_size, max_num_keypoints, 1))
        batched_match_idxs1 = torch.reshape(batched_match_idxs1, (batch_size, 1, max_num_keypoints))
        match_correspondence = batched_match_idxs0 - batched_match_idxs1
        s_idxs = torch.nonzero(match_correspondence == 0, as_tuple=True)
        # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) corresponds to
        # point (h', w') and 0 otherwise

    # Normalize and filter descriptors to only use the ones at keypoints
    descriptors0 = torch.nn.functional.normalize(descriptors0, dim=1)
    descriptors1 = torch.nn.functional.normalize(descriptors1, dim=1)
    eval_descriptors0 = descriptors0.permute(0, 2, 3, 1)[kp_idxs0]
    eval_descriptors1 = descriptors1.permute(0, 2, 3, 1)[kp_idxs1]

    # Calculate loss
    loss_batch = [double_softmax_distance(eval_descriptors0[cum_matches_until_sample0[batch]:cum_matches_until_sample0[batch+1]],
                                      eval_descriptors1[cum_matches_until_sample1[batch]:cum_matches_until_sample1[batch+1]],
                                      config["double_softmax_loss_temperature"])
                                      [s_idxs[1][s_idxs[0]==batch], s_idxs[2][s_idxs[0]==batch]]  # select roundtrips in the tensor of distances
                    for batch in range(batch_size)]
    normalization = max(len(s_idxs[0]), 1)
    return torch.sum(torch.cat(loss_batch)) / normalization

def feature_cossim_loss(features, gt_features, config, valid_mask=None):
    gt_features = gt_features.detach()

    # Normalize features to compute cosine similarity
    s_feat_norm = F.normalize(features, dim=1)
    t_feat_norm = F.normalize(gt_features, dim=1)

    if valid_mask is not None:
        valid_mask = F.max_pool2d(valid_mask.float(), kernel_size=config["grid_size"], stride=config["grid_size"])

    # Compute cosine similarity
    cosine_sim = torch.sum(
        s_feat_norm * t_feat_norm, dim=1
    )  # Shape: (batch_size, num_tokens)

    # Mask tokens with cosine similarity > alpha
    mask = (cosine_sim <= config["cossim_loss_alpha"]) & (cosine_sim >= config["cossim_loss_beta"])

    if valid_mask is not None:
        mask = mask & (valid_mask.sum(1) > 0)

    # Compute cosine distance for the valid tokens
    valid_diff = (
        1.0 - cosine_sim[mask]
    )  # Cosine distance is 1 - cosine similarity

    # Accumulate loss and count valid tokens
    if valid_diff.numel() > 0:
        feature_loss = valid_diff.mean()
        return feature_loss 
    
    else:
        return torch.tensor(0.0, device=features.device, requires_grad=True)


def feature_l2_loss(features, gt_features, valid_mask=None):
    """
    适用于中间层特征 (Feature Consistency Loss)
    不仅对齐方向，还对齐强度 (Sharpness/Magnitude)
    """
    gt_features = gt_features.detach()

    # 直接计算差值的 L2 范数 (MSE)
    diff = (features - gt_features) ** 2
    
    # 在 Channel 维度求和 -> (B, H, W)
    diff = diff.sum(dim=1) 

    if valid_mask is not None:
        if valid_mask.shape[-2:] != features.shape[-2:]:
            valid_mask = F.interpolate(valid_mask.float(), size=features.shape[-2:], mode='nearest')
        
        # 应用 mask
        mask = valid_mask.squeeze(1) > 0
        valid_diff = diff[mask]
    else:
        valid_diff = diff

    if valid_diff.numel() > 0:
        return valid_diff.mean()
    else:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

def masked_l1_loss(input, target, config, mask):  
    if mask is not None:
        mask = F.max_pool2d(mask.float(), kernel_size=config["grid_size"], stride=config["grid_size"])[:,0,:,:]
    diff = torch.abs(input - target)
    masked_diff = diff * mask
    masked_diff = masked_diff.mean(dim=1)  # mean over channel dimension
    loss = masked_diff.sum() / mask.sum().clamp(min=1.0)
    return loss

def masked_mse_loss(input, target, config, mask):
    diff = (input - target) ** 2
    masked_diff = diff * mask
    loss = masked_diff.sum() / mask.sum().clamp(min=1.0)
    return loss


def logits_loss(logits, gt_logits, config, mask=None):
    """
    Args:
        logits (torch.Tensor): Student prediction [B, 65, Hc, Wc]
        gt_logits (torch.Tensor): Teacher target [B, 65, Hc, Wc]
        config (dict): Configuration dictionary
        mask (torch.Tensor): Optional mask [B, 1, H, W] (original resolution)
    """
    pred_keypoints_logits = logits
    gt_keypoints_logits = gt_logits

    assert (
        pred_keypoints_logits.shape == gt_keypoints_logits.shape
    ), f"pred: {pred_keypoints_logits.shape}, gt: {gt_keypoints_logits.shape}"

    channel_dim = pred_keypoints_logits.shape[1]
    assert (
        channel_dim == config["grid_size"] * config["grid_size"] + 1
    ), f"channel_dim: {channel_dim}, cell_size: {config['grid_size']}"

    mode = config.get("logits_loss_mode", "mse")
    loss = None
    if mode == "kl_div":
        # 1. 温度系数平滑 (Temperature Scaling)
        # T > 1 会让分布更平滑，有助于学生学习老师的暗知识(dark knowledge)
        T = config.get("distillation_temperature", 1.0)
        
        # 2. 计算概率分布
        # 注意：要在 Channel 维度 (dim=1) 进行 Softmax，保留 Dustbin 通道以形成完整概率分布
        # Student 使用 LogSoftmax (KLDivLoss 的要求)
        student_log_prob = F.log_softmax(pred_keypoints_logits / T, dim=1)
        # Teacher 使用 Softmax
        teacher_prob = F.softmax(gt_keypoints_logits / T, dim=1)

        # 3. 计算 KL Divergence
        # reduction='none' 保留维度以便应用 Mask
        # log_target=False 表示 target 是普通概率而非 log 概率
        loss = F.kl_div(student_log_prob, teacher_prob, reduction='none', log_target=False)
        
        # 4. 在 Channel 维度求和 (T * T 是为了梯度缩放平衡，但在这种 loss 里通常省略或设为 1)
        # 结果形状: [B, Hc, Wc]
        loss = loss.sum(dim=1) 

        # 5. 处理 Mask
        if mask is not None:
            # Mask 通常是全分辨率 [B, 1, H, W]，需要下采样匹配 Feature Map [B, Hc, Wc]
            if mask.shape[-2:] != loss.shape[-2:]:
                # 使用 max_pool 确保只要 8x8 grid 内有一个坏点就被 mask (或者用 nearest)
                # mask shape: [B, 1, H, W] -> [B, Hc, Wc]
                mask_grid = F.interpolate(mask.float(), size=loss.shape[-2:], mode='nearest').squeeze(1)
            else:
                mask_grid = mask.squeeze(1)
            
            # 只计算有效区域的平均 Loss
            loss = (loss * mask_grid).sum() / (mask_grid.sum() + 1e-6)
        else:
            loss = loss.mean()
        
    elif mode == "mse":
        if config["grid_size"] > 1:
            # remove the last (dustbin) cell from the list
            pred_keypoints_logits, _ = pred_keypoints_logits.split(
                config["grid_size"] * config["grid_size"], dim=1
            )
            gt_keypoints_logits, _ = gt_keypoints_logits.split(
                config["grid_size"] * config["grid_size"], dim=1
            )

            # change the dimensions to get an output shape of (batch_size, H, W)
            pred_keypoints_logits = F.pixel_shuffle(
                pred_keypoints_logits, config["grid_size"]
            )
            gt_keypoints_logits = F.pixel_shuffle(gt_keypoints_logits, config["grid_size"])

        else:
            assert (
                channel_dim == 1
            ), f"channel_dim: {channel_dim}, cell_size: {config['grid_size']}"

        pred_keypoints_logits = pred_keypoints_logits.view(
            pred_keypoints_logits.shape[0], -1
        )
        gt_keypoints_logits = gt_keypoints_logits.view(gt_keypoints_logits.shape[0], -1)
        
        if mask is not None:
            mask = mask.view(mask.shape[0], -1)

        loss = F.mse_loss(pred_keypoints_logits, gt_keypoints_logits, reduction="none")
        
        if mask is not None:
            loss = loss * mask
            
        loss = loss.mean()

    return loss

def descriptor_feature_loss(descriptors, gt_descriptors, config, use_mask=False, mask=None,):

    if not use_mask:
        mask = None

    if mask is not None and mask.shape[1] == 1:
        mask = mask.repeat(1, descriptors.shape[1], 1, 1)

    assert descriptors.shape == gt_descriptors.shape
    
    mode = config.get("descriptor_feature_loss_mode", "mse")

    if mode == "mse":
        # loss = (torch.norm(pred_descriptors - gt_descriptors, dim=1) * mask).sum() / mask.sum()
        if mask is not None:
            if mask.sum() == 0:
                loss = torch.tensor(
                    0.0,
                    dtype=descriptors.dtype,
                    device=descriptors.device,
                )
            loss = (
                ((descriptors - gt_descriptors) ** 2) * mask
            ).sum() / mask.sum()
        else:
            loss = F.mse_loss(descriptors, gt_descriptors)
    elif mode == "mae":
        if mask is not None:
            if mask.sum() == 0:
                loss = torch.tensor(
                    0.0,
                    dtype=descriptors.dtype,
                    device=descriptors.device,
                )
            loss = (
                torch.abs(descriptors - gt_descriptors) * mask
            ).sum() / mask.sum()
        else:
            loss = F.l1_loss(descriptors, gt_descriptors)
    elif mode == "cosine_similarity":
        if mask is not None:
            if mask.sum() == 0:
                loss = torch.tensor(
                    0.0,
                    dtype=descriptors.dtype,
                    device=descriptors.device,
                )
            cosine_sim = F.cosine_similarity(
                descriptors, gt_descriptors, dim=1
            )
            loss = 1 - cosine_sim.view(-1)[mask.view(-1)].mean()
        else:
            cosine_sim = F.cosine_similarity(
                descriptors, gt_descriptors, dim=1
            )
            loss = 1 - cosine_sim.mean()

    return loss



def create_meshgrid(height, width, device):
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    return torch.stack([x, y], dim=0).float() # [2, H, W]

def subpixel_coordinate_loss(logits, gt_keypoint_map, window_size=11, concentration_weight=0.1):
    """
    向量化版本的积分回归 Loss，无 Python 循环，速度极快。
    
    Args:
        logits: [B, 65, H/8, W/8] - 原始网络输出
        gt_keypoint_map: [B, H, W] - 关键点 mask (0 or 1)
        window_size: 窗口大小，建议 7
    """
    device = logits.device
    B, _, Hc, Wc = logits.shape
    
    # 1. PixelShuffle 还原全分辨率 Heatmap
    # [B, 64, H/8, W/8] -> [B, H, W]
    logits_nodust = logits[:, :-1, :, :]
    heatmap = F.pixel_shuffle(logits_nodust, 8).squeeze(1)
    H, W = heatmap.shape[-2:]

    # 2. 提取所有 GT 关键点的坐标 (batch_idx, y, x)
    # 这一步代替了外层的 for b in range(batch_size)
    # b_idxs, ys, xs 都是一维向量，长度为 N (总关键点数)
    b_idxs, ys, xs = torch.where(gt_keypoint_map > 0)
    
    if len(b_idxs) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 3. 构建局部窗口的偏移量网格
    # 例如 window_size=7, radius=3, offsets范围 [-3, ..., 3]
    radius = window_size // 2
    local_grid_y, local_grid_x = torch.meshgrid(
        torch.arange(-radius, radius+1, device=device),
        torch.arange(-radius, radius+1, device=device),
        indexing='ij'
    )
    
    # Flatten 网格: [window_size^2] -> [49]
    offset_y = local_grid_y.flatten()
    offset_x = local_grid_x.flatten()

    # 4. 广播计算所有采样点的绝对坐标
    # [N, 1] + [49] -> [N, 49]
    # sample_y[i, j] 表示第 i 个关键点的窗口中第 j 个像素的绝对 y 坐标
    sample_y = ys.unsqueeze(1) + offset_y.unsqueeze(0)
    sample_x = xs.unsqueeze(1) + offset_x.unsqueeze(0)
    sample_b = b_idxs.unsqueeze(1).expand_as(sample_y)

    # 5. 边界掩码 (Masking)
    # 标记出超出图像边界的采样点
    valid_mask = (sample_y >= 0) & (sample_y < H) & \
                 (sample_x >= 0) & (sample_x < W)

    # 6. 安全采样 (Safe Gathering)
    # 为了防止 index out of bounds 报错，先把越界的坐标 clamp 到图像内
    # 然后利用 valid_mask 把这些假数据处理掉
    safe_y = sample_y.clamp(0, H - 1)
    safe_x = sample_x.clamp(0, W - 1)

    # 一次性取出所有 patches: [N, 49]
    # 这是最快的一步，直接显存拷贝
    patches = heatmap[sample_b, safe_y, safe_x]

    # 7. 处理无效区域
    # 对于越界的点，将其 Logits 设为极小值 (-inf)，这样 Softmax 后概率为 0
    # 注意：patches 是视图，需 clone 或直接计算
    patches = patches.masked_fill(~valid_mask, -1e9)

    # 8. 空间 Softmax
    # [N, 49] -> [N, 49] (概率分布)
    probs = F.softmax(patches, dim=1)

    # 9. 积分回归计算重心 (Expectation)
    # 我们希望重心在局部窗口的中心，即 offset 为 0
    # sum(prob * offset)
    # [N, 49] * [1, 49] -> sum -> [N]
    pred_offset_x = torch.sum(probs * offset_x.unsqueeze(0), dim=1)
    pred_offset_y = torch.sum(probs * offset_y.unsqueeze(0), dim=1)
    
    position_loss = torch.abs(pred_offset_x) + torch.abs(pred_offset_y)

    # 惩罚概率分布的扩散程度。我们希望概率主要集中在 offset=0 的地方。
    # 计算公式：Sum( P * dist^2 )
    # 意思就是：离中心越远的像素，如果有概率值，惩罚就越重（平方级）。
    
    # 距离中心的平方距离网格
    dist_sq_x = offset_x ** 2
    dist_sq_y = offset_y ** 2
    
    # 期望距离平方 (Expected Squared Distance from Center)
    variance_x = torch.sum(probs * dist_sq_x.unsqueeze(0), dim=1)
    variance_y = torch.sum(probs * dist_sq_y.unsqueeze(0), dim=1)
    
    spread_loss = variance_x + variance_y
    # =========================================================================

    total_loss = position_loss.mean() + (concentration_weight * spread_loss.mean())

    return total_loss


######
def super_event_loss(logits0, logits1, descriptors0, descriptors1, ground_truth_keypoint_map0, ground_truth_keypoint_map1, config):
    det_loss0 = detector_loss(logits0, ground_truth_keypoint_map0, config)
    det_loss1 = detector_loss(logits1, ground_truth_keypoint_map1, config)
    if config["pixel_wise_predictions"]:
        desc_loss = double_softmax_loss(descriptors0, descriptors1, ground_truth_keypoint_map0, ground_truth_keypoint_map1, config)
        pos_loss = neg_loss = 0
    else:
        desc_loss, pos_loss, neg_loss = descriptor_loss(descriptors0, descriptors1, ground_truth_keypoint_map0, ground_truth_keypoint_map1, config)

    loss = (det_loss0 + det_loss1 + config["lambda_loss"] * desc_loss)
    return loss, det_loss0, det_loss1, config["lambda_loss"] * desc_loss, config["lambda_loss"] * pos_loss, config["lambda_loss"] * neg_loss


def super_event_loss_fusion(input_group, config):
    if "event_count_mask" in config and config["event_count_mask"]:
        det_loss0 = detector_loss(input_group["logits0"], input_group["ground_truth_keypoint_map0"], 
                                  config, input_group['mask_E0_exsits'])
        det_loss1 = detector_loss(input_group["logits1"], input_group["ground_truth_keypoint_map1"], 
                                  config, input_group['mask_E1_exsits'])
    elif "event_points_mask" in config and config["event_points_mask"]:
        det_loss0 = detector_loss_event_mask(input_group["logits0"], input_group["ground_truth_keypoint_map0"], 
                                  config, input_group['E_points_2d_0'])
        det_loss1 = detector_loss_event_mask(input_group["logits1"], input_group["ground_truth_keypoint_map1"], 
                                  config, input_group['E_points_2d_1'])
    else:
        det_loss0 = detector_loss(input_group["logits0"], input_group["ground_truth_keypoint_map0"], config)
        det_loss1 = detector_loss(input_group["logits1"], input_group["ground_truth_keypoint_map1"], config)

    if config.get("event_points_mask") and config.get("masked_descriptor_loss"):
        desc_loss, pos_loss, neg_loss = descriptor_loss(input_group["descriptors0"], input_group["descriptors1"], 
                                    input_group["ground_truth_keypoint_map0"], input_group["ground_truth_keypoint_map1"],
                                    config, input_group['E_points_2d_0'], input_group['E_points_2d_1'])
    elif config.get("double_softmax_loss", False):

        # ================== [新增] 自动上采样逻辑 ==================
        # 目的：确保输入 descriptor_loss 的特征图尺寸与 Keypoint Map (H, W) 一致
        # 这样做不会影响上面可能用到的原始小尺寸特征 loss
        
        target_h, target_w = input_group["ground_truth_keypoint_map0"].shape[-2:]
        desc0_upsample = input_group["descriptors0"]
        desc1_upsample = input_group["descriptors1"]

        # 检查尺寸，如果不一致则上采样
        if desc0_upsample.shape[-2:] != (target_h, target_w):
            desc0_upsample = F.interpolate(desc0_upsample, size=(target_h, target_w), mode='bilinear', align_corners=False)
            desc1_upsample = F.interpolate(desc1_upsample, size=(target_h, target_w), mode='bilinear', align_corners=False)
        # =========================================================

        desc_loss = double_softmax_loss(desc0_upsample, desc1_upsample, 
                                    input_group["ground_truth_keypoint_map0"], input_group["ground_truth_keypoint_map1"],
                                    config)
        pos_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
        neg_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    else:
        desc_loss, pos_loss, neg_loss = descriptor_loss(input_group["descriptors0"], input_group["descriptors1"], 
                                    input_group["ground_truth_keypoint_map0"], input_group["ground_truth_keypoint_map1"],
                                    config)

    # desc_loss, pos_loss, neg_loss = descriptor_loss(input_group["descriptors0"], input_group["descriptors1"], 
    #                                 input_group["ground_truth_keypoint_map0"], input_group["ground_truth_keypoint_map1"],
    #                                 config)


    ## hard area detection loss
    det_loss_hard_E = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    det_loss_hard_F = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    if config.get("hard_E_detector_loss"):
        hard_mask_E0 = ~input_group['mask_E0'].bool() & input_group['mask_F0'].bool()
        hard_mask_E1 = ~input_group['mask_E1'].bool() & input_group['mask_F1'].bool()
        det_loss_hard_E = detector_loss(input_group["logits0"], input_group["ground_truth_keypoint_map0"], config, hard_mask_F0) + \
                detector_loss(input_group["logits1"], input_group["ground_truth_keypoint_map1"], config, hard_mask_F1)
    if config.get("hard_F_detector_loss"):
        hard_mask_F0 = ~input_group['mask_F0'].bool() & input_group['mask_E0'].bool()
        hard_mask_F1 = ~input_group['mask_F1'].bool() & input_group['mask_E1'].bool()
        det_loss_hard_F = detector_loss(input_group["logits0"], input_group["ground_truth_keypoint_map0"], config, hard_mask_E0) + \
                        detector_loss(input_group["logits1"], input_group["ground_truth_keypoint_map1"], config, hard_mask_E1)

    e_desc_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    if config.get("event_desc_loss"):
        E_keypoints_nms0 = input_group["E_keypoints_nms_list_0"][0]
        E_keypoints_nms1 = input_group["E_keypoints_nms_list_1"][0]
        E_desc_nms0 = input_group["E_desc_nms_list_0"][0]
        E_desc_nms1 = input_group["E_desc_nms_list_1"][0]

        E_matches_index = batch_mutual_nearest_neighbor(E_desc_nms0.unsqueeze(0), E_desc_nms1.unsqueeze(0))
        E_matches = [E_keypoints_nms0[E_matches_index[0][:,0]], 
                     E_keypoints_nms1[E_matches_index[0][:,1]]]
        E_kp_map0, E_kp_map1 = matches2keypoint_maps(E_matches, input_group["ground_truth_keypoint_map0"].shape[1:])
        e_desc_loss, _, _ = descriptor_loss(input_group["descriptors0"], input_group["descriptors1"], 
                                    E_kp_map0.unsqueeze(0), E_kp_map1.unsqueeze(0),
                                    config)

    feature_lossF_hard = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    feature_lossF = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    feature_lossE_hard = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    feature_lossE = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    if config.get("feature_loss"): 
        if config.get("feature_loss_type") == "F&E": # F&E, sp
            feature_lossE = (feature_cossim_loss(input_group["feature_0"], input_group["feature_E0"], config, input_group['mask_E0']) + 
                            feature_cossim_loss(input_group["feature_1"], input_group["feature_E1"], config, input_group['mask_E1']))/2
            feature_lossF = (feature_cossim_loss(input_group["feature_0"], input_group["feature_F0"], config, input_group['mask_F0']) + 
                            feature_cossim_loss(input_group["feature_1"], input_group["feature_F1"], config, input_group['mask_F1']))/2
            
            feature_lossE_hard = (feature_cossim_loss(input_group["feature_0"], input_group["feature_E0"], config, hard_mask_F0) + 
                                feature_cossim_loss(input_group["feature_1"], input_group["feature_E1"], config, hard_mask_F1))/2
            feature_lossF_hard = (feature_cossim_loss(input_group["feature_0"], input_group["feature_F0"], config, hard_mask_E0) + 
                              feature_cossim_loss(input_group["feature_1"], input_group["feature_F1"], config, hard_mask_E1))/2
        else:
            # feature_loss0 = feature_cossim_loss(input_group["feature_0"], input_group["feature_sp0"], config)
            # feature_loss1 = feature_cossim_loss(input_group["feature_1"], input_group["feature_sp1"], config)
            feature_loss0 = feature_l2_loss(input_group["feature_0"], input_group["feature_sp0"])
            feature_loss1 = feature_l2_loss(input_group["feature_1"], input_group["feature_sp1"])
            feature_lossE = feature_loss0 + feature_loss1

    desc_l1_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    desc_l1_loss_hard = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    if config.get("desc_l1_loss", False):
        if config["train_type"] == "event":
            l1_E = masked_l1_loss(F.normalize(input_group["descriptors0"],dim=1), input_group["gt_descriptors0"].detach(), config, input_group['mask_E0']) +\
                masked_l1_loss(F.normalize(input_group["descriptors1"],dim=1), input_group["gt_descriptors1"].detach(), config, input_group['mask_E1'])

            desc_l1_loss = l1_E
        
        elif config["train_type"] == "fusion":
            l1_F = masked_l1_loss(F.normalize(input_group["descriptors0"],dim=1), input_group["F_desc0"].detach(), config, input_group['mask_F0']) +\
                masked_l1_loss(F.normalize(input_group["descriptors1"],dim=1), input_group["F_desc1"].detach(), config, input_group['mask_F1'])
            l1_E = masked_l1_loss(F.normalize(input_group["descriptors0"],dim=1), input_group["E_desc0"].detach(), config, input_group['mask_E0']) +\
                masked_l1_loss(F.normalize(input_group["descriptors1"],dim=1), input_group["E_desc1"].detach(), config, input_group['mask_E1'])
            if config.get("pretrained_fe_detector"):
                l1_E_hard = masked_l1_loss(F.normalize(input_group["descriptors0"],dim=1), input_group["E_desc0"].detach(), config, hard_mask_F0) +\
                            masked_l1_loss(F.normalize(input_group["descriptors1"],dim=1), input_group["E_desc1"].detach(), config, hard_mask_F1)
                l1_F_hard = masked_l1_loss(F.normalize(input_group["descriptors0"],dim=1), input_group["F_desc0"].detach(), config, hard_mask_E0) +\
                            masked_l1_loss(F.normalize(input_group["descriptors1"],dim=1), input_group["F_desc1"].detach(), config, hard_mask_E1)
            
            if config["backbone"] in ["vgg-mask-sp"]:
                l1_F_hard = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
                l1_F = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)

            desc_l1_loss = (l1_F + l1_E)/2
            desc_l1_loss_hard = (l1_F_hard + l1_E_hard)/2
    elif config.get("descriptor_feature_loss", False):
        desc_type = config.get("desc_type", "raw")
        if desc_type == "raw":
            desc_fea0 = input_group["descriptors0"]
            desc_fea1 = input_group["descriptors1"]
            gt_desc_fea0 = input_group["gt_descriptors0"]
            gt_desc_fea1 = input_group["gt_descriptors1"]
        elif desc_type == "normalized":
            desc_fea0 = input_group["descriptors0_expanded"]
            desc_fea1 = input_group["descriptors1_expanded"]
            gt_desc_fea0 = input_group["gt_descriptors0_expanded"]
            gt_desc_fea1 = input_group["gt_descriptors1_expanded"]

        desc_fea_loss = descriptor_feature_loss(desc_fea0, gt_desc_fea0, config) + \
                    descriptor_feature_loss(desc_fea1, gt_desc_fea1, config)
        desc_l1_loss = desc_fea_loss

    keypoint_mse_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    if config.get("pretrained_fe_detector") and config.get("keypoint_mse_loss"):
        F_prob0 = input_group["F_prob0"].detach()
        F_prob1 = input_group["F_prob1"].detach()
        E_prob0 = input_group["E_prob0"].detach()
        E_prob1 = input_group["E_prob1"].detach()

        mse_F = masked_mse_loss(input_group["prob0"], F_prob0, config, input_group['mask_F0']) + \
                masked_mse_loss(input_group["prob1"], F_prob1, config, input_group['mask_F1'])
        mse_E = masked_mse_loss(input_group["prob0"], E_prob0, config, input_group['mask_E0']) + \
                masked_mse_loss(input_group["prob1"], E_prob1, config, input_group['mask_E1'])
        
        if config["backbone"] in ["vgg-mask-sp"]:
            mse_F = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
        
        keypoint_mse_loss = (mse_F + mse_E)/2
    elif config.get("logits_loss",False):
        det_loss0 = logits_loss(input_group["logits0"], input_group["gt_logits0"], config)
        det_loss1 = logits_loss(input_group["logits1"], input_group["gt_logits1"], config)
        det_logits_loss = det_loss0 + det_loss1
        keypoint_mse_loss = det_logits_loss
    elif config.get("coordinate_loss", False):
        coord_loss0 = subpixel_coordinate_loss(input_group["logits0"], input_group["ground_truth_keypoint_map0"], window_size=7, concentration_weight=config.get("coordinate_loss_concentration_weight", 0.1))
        coord_loss1 = subpixel_coordinate_loss(input_group["logits1"], input_group["ground_truth_keypoint_map1"], window_size=7, concentration_weight=config.get("coordinate_loss_concentration_weight", 0.1))
        det_logits_loss = coord_loss0 + coord_loss1
        keypoint_mse_loss = det_logits_loss

    loss = (det_loss0 + det_loss1 + 
            config["lambda_cossim"]*feature_lossE + 
            config["lambda_cossim"]*feature_lossF + 
            config["lambda_loss"] * desc_loss + 
            # config["lambda_l1"] * keypoint_mse_loss +
            config.get("lambda_logits_loss", 0) * keypoint_mse_loss +
            config["lambda_l1"] * desc_l1_loss+
            det_loss_hard_E * config["lambda_hard_det_loss"] +
            det_loss_hard_F * config["lambda_hard_det_loss"] +
            config["lambda_hard_cossim"]*feature_lossE_hard +
            config["lambda_hard_cossim"]*feature_lossF_hard +
            desc_l1_loss_hard * config["lambda_hard_l1"]+
            config["lambda_e_loss"] * e_desc_loss)
        
    
    loss_dict = {
        "loss": loss,
        "det_loss0": det_loss0,
        "det_loss1": det_loss1, 
        "feature_loss0": config["lambda_cossim"]* feature_lossE,
        "feature_loss1": config["lambda_cossim"]* feature_lossF,
        "desc_l1_loss": config["lambda_l1"] * desc_l1_loss,
        "keypoint_mse_loss": config["lambda_l1"] * keypoint_mse_loss,
        "desc_loss": config["lambda_loss"] * desc_loss,
        "pos_loss": config["lambda_loss"] * pos_loss,
        "neg_loss": config["lambda_loss"] * neg_loss,

        "feature_lossE_hard": config["lambda_hard_cossim"] * feature_lossE_hard,
        "feature_lossF_hard": config["lambda_hard_cossim"] * feature_lossF_hard,
        "det_loss_hard_E": det_loss_hard_E * config["lambda_hard_det_loss"],
        "det_loss_hard_F": det_loss_hard_F * config["lambda_hard_det_loss"],
        "desc_l1_loss_hard": desc_l1_loss_hard * config["lambda_hard_l1"],
        "e_desc_loss": config["lambda_e_loss"] * e_desc_loss,

    }

    return loss_dict


def super_event_loss_fusion_deblur(input_group, config):
    det_loss0 = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    det_loss1 = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    det_logits_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    if config.get("logits_loss",False):
        det_loss0 = logits_loss(input_group["logits0"], input_group["gt_logits0"], config)
        det_loss1 = logits_loss(input_group["logits1"], input_group["gt_logits1"], config)
        det_logits_loss = det_loss0 + det_loss1
    elif config.get("coordinate_loss", False):
        coord_loss0 = subpixel_coordinate_loss(input_group["logits0"], input_group["ground_truth_keypoint_map0"], window_size=7, concentration_weight=config.get("coordinate_loss_concentration_weight", 0.1))
        coord_loss1 = subpixel_coordinate_loss(input_group["logits1"], input_group["ground_truth_keypoint_map1"], window_size=7, concentration_weight=config.get("coordinate_loss_concentration_weight", 0.1))
        det_logits_loss = coord_loss0 + coord_loss1

    if "event_count_mask" in config and config["event_count_mask"]:
        det_loss0 = detector_loss(input_group["logits0"], input_group["ground_truth_keypoint_map0"], 
                                config, input_group['mask_E0_exsits'])
        det_loss1 = detector_loss(input_group["logits1"], input_group["ground_truth_keypoint_map1"], 
                                config, input_group['mask_E1_exsits'])
    elif "event_points_mask" in config and config["event_points_mask"]:
        det_loss0 = detector_loss_event_mask(input_group["logits0"], input_group["ground_truth_keypoint_map0"], 
                                config, input_group['E_points_2d_0'])
        det_loss1 = detector_loss_event_mask(input_group["logits1"], input_group["ground_truth_keypoint_map1"], 
                                config, input_group['E_points_2d_1'])
    else:
        det_loss0 = detector_loss(input_group["logits0"], input_group["ground_truth_keypoint_map0"], config)
        det_loss1 = detector_loss(input_group["logits1"], input_group["ground_truth_keypoint_map1"], config)


    desc_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    pos_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    neg_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    desc_fea_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    if config.get("descriptor_feature_loss", False):
        desc_type = config.get("desc_type", "raw")
        if desc_type == "raw":
            desc_fea0 = input_group["descriptors0"]
            desc_fea1 = input_group["descriptors1"]
            gt_desc_fea0 = input_group["gt_descriptors0"]
            gt_desc_fea1 = input_group["gt_descriptors1"]
        elif desc_type == "normalized":
            desc_fea0 = input_group["descriptors0_expanded"]
            desc_fea1 = input_group["descriptors1_expanded"]
            gt_desc_fea0 = input_group["gt_descriptors0_expanded"]
            gt_desc_fea1 = input_group["gt_descriptors1_expanded"]

        desc_fea_loss = descriptor_feature_loss(desc_fea0, gt_desc_fea0, config) + \
                    descriptor_feature_loss(desc_fea1, gt_desc_fea1, config)

    if config.get("clear_blur_matching", {}).get("enable", False):
        desc_loss,pos_loss,neg_loss = dense_descriptor_loss(
                                        input_group["descriptors0"], 
                                        input_group["descriptors1"], 
                                        input_group["pixel_mapping"],
                                        config)
    elif config.get("event_points_mask") and config.get("masked_descriptor_loss"):
        desc_loss, pos_loss, neg_loss = descriptor_loss(input_group["descriptors0"], input_group["descriptors1"], 
                                    input_group["ground_truth_keypoint_map0"], input_group["ground_truth_keypoint_map1"],
                                    config, input_group['E_points_2d_0'], input_group['E_points_2d_1'])
    elif config.get("double_softmax_loss", False):

        # ================== [新增] 自动上采样逻辑 ==================
        # 目的：确保输入 descriptor_loss 的特征图尺寸与 Keypoint Map (H, W) 一致
        # 这样做不会影响上面可能用到的原始小尺寸特征 loss
        
        target_h, target_w = input_group["ground_truth_keypoint_map0"].shape[-2:]
        desc0_upsample = input_group["descriptors0"]
        desc1_upsample = input_group["descriptors1"]

        # 检查尺寸，如果不一致则上采样
        if desc0_upsample.shape[-2:] != (target_h, target_w):
            desc0_upsample = F.interpolate(desc0_upsample, size=(target_h, target_w), mode='bilinear', align_corners=False)
            desc1_upsample = F.interpolate(desc1_upsample, size=(target_h, target_w), mode='bilinear', align_corners=False)
        # =========================================================

        desc_loss = double_softmax_loss(desc0_upsample, desc1_upsample, 
                                    input_group["ground_truth_keypoint_map0"], input_group["ground_truth_keypoint_map1"],
                                    config)
        pos_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
        neg_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    else:
        desc_loss, pos_loss, neg_loss = descriptor_loss(input_group["descriptors0"], input_group["descriptors1"], 
                                    input_group["ground_truth_keypoint_map0"], input_group["ground_truth_keypoint_map1"],
                                    config)
    


    feature_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    if config.get("feature_loss"): 
        if config.get("use_seprate_decoder", False):
            feature_loss0_det = feature_l2_loss(input_group["feature_0_detector"], input_group["feature_sp0"])
            feature_loss1_det = feature_l2_loss(input_group["feature_1_detector"], input_group["feature_sp1"])
            feature_loss0_desc = feature_l2_loss(input_group["feature_0_desc"], input_group["feature_sp0"])
            feature_loss1_desc = feature_l2_loss(input_group["feature_1_desc"], input_group["feature_sp1"])
            feature_loss = feature_loss0_det + feature_loss1_det + feature_loss0_desc + feature_loss1_desc
        else: 
            feature_loss0 = feature_l2_loss(input_group["feature_0"], input_group["feature_sp0"])
            feature_loss1 = feature_l2_loss(input_group["feature_1"], input_group["feature_sp1"])
            feature_loss = feature_loss0 + feature_loss1

        if config.get("feature_loss_for_event", False):
            # mask0 = input_group['mask_E0_exsits']
            # mask1 = input_group['mask_E1_exsits']
            # target_size = input_group["feature_e0"].shape[-2:]
            
            # # 使用 interpolate 最近邻插值 (nearest) 保持 0/1 性质
            # mask0 = F.interpolate(mask0.float(), size=target_size, mode='nearest')
            # mask1 = F.interpolate(mask1.float(), size=target_size, mode='nearest')

            # feature_loss0_e = feature_l2_loss(input_group["feature_e0"], input_group["feature_sp0"], mask0)
            # feature_loss1_e = feature_l2_loss(input_group["feature_e1"], input_group["feature_sp1"], mask1)
            
            feature_loss0_e = feature_l2_loss(input_group["feature_e0"], input_group["feature_sp0"])
            feature_loss1_e = feature_l2_loss(input_group["feature_e1"], input_group["feature_sp1"])

            feature_loss0_f = feature_l2_loss(input_group["feature_f0"], input_group["feature_sp0"])
            feature_loss1_f = feature_l2_loss(input_group["feature_f1"], input_group["feature_sp1"])
            
            feature_loss += feature_loss0_e + feature_loss1_e + feature_loss0_f + feature_loss1_f



    desc_l1_sparsity_loss = torch.tensor(0.0, device=input_group["logits0"].device, requires_grad=True)
    if config.get("desc_l1_sparsity_loss"):
        desc_l1_sparsity_loss = (torch.mean(torch.abs(input_group["descriptors0"])) + 
                                 torch.mean(torch.abs(input_group["descriptors1"]))) / 2

    
    loss = (config["lambda_det_loss"] * det_loss0 + config["lambda_det_loss"] * det_loss1 + 
            config["lambda_loss"] * desc_loss + 
            config["lambda_cossim"] * feature_loss +
            config["lambda_l1_sparsity"] * desc_l1_sparsity_loss+
            config.get("lambda_logits_loss", 0.0) * det_logits_loss +
            config.get("lambda_descriptor_feature_loss", 0.0) * desc_fea_loss)
    
    loss_dict = {
        "loss": loss,
        "det_loss0": config["lambda_det_loss"] * det_loss0,
        "det_loss1": config["lambda_det_loss"] * det_loss1, 
        "desc_loss": config["lambda_loss"] * desc_loss,
        "pos_loss": config["lambda_loss"] * pos_loss,
        "neg_loss": config["lambda_loss"] * neg_loss,
        "feature_loss": config["lambda_cossim"]* feature_loss,
        "desc_l1_sparsity_loss": config["lambda_l1_sparsity"] * desc_l1_sparsity_loss,
        "det_logits_loss": config.get("lambda_logits_loss", 0.0) * det_logits_loss,
        "desc_fea_loss": config.get("lambda_descriptor_feature_loss", 0.0) * desc_fea_loss,
    }
    return loss_dict
