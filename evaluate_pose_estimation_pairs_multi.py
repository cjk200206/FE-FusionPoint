import argparse
import sys, io
import cv2
from glob import glob
import hdf5plugin
import math
import numpy as np
import os
import itertools
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from pandas import read_csv
from scipy.spatial.transform import Rotation as R
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
from typing import Any, Dict, Tuple
import h5py


from data.util import create_tencode_from_ts
from data_preparation.util import helpers
from models.super_event_fusion import SuperEventFusion
from models.super_event import SuperEvent
from models.superpoint import SuperPointBatchNormed

from models.util import fast_nms
from ts_generation.generate_ts import TsGenerator
from util import visualization
from util.eval_utils import fix_seed
from util.Interpolator import PoseInterpolator, PoseInterpolatorSpline


def predict_keypoints(args, config, ts_shape, model, events, image_data, 
                      start_time, end_time, cropped_shape=None, compiled=False):
    if cropped_shape == None:
        cropped_shape = ts_shape
    settings = {"shape": ts_shape, "delta_t": args.model_delta_t}
    ts_gen = TsGenerator(settings=settings, device=device)

    # Create pose interpolator
    
    pose_interpolator = image_data["pose_interpolator"]
    frame_timestamps = image_data["image_ts"]
    frames = image_data["image"]


    if args.dataset_name in ["ecd", "dvs"]:
        events = torch.from_numpy(events).to(device)
        events_t = events[:, 0].clone()
    elif args.dataset_name == "eds":
        # Load timestamps into numpy array to speed-up loop
        events_t = np.array(events['t'], dtype=np.int64)
    elif args.dataset_name == "mvsec":
        events = torch.from_numpy(events[:]).to(device)
        events_t = events[:, 2].clone()

        # events_t = np.array(events[:,2]) #[x,y,t,p]

    print("Starting event loop.")
    current_event_idx = 0
    pred_dict = {}
    ts_vis_dict = {}
    frame_vis_dict = {}

    # Time measurement
    model_inference_time = 0.
    model_inference_iterations = 0
    nms_time = 0.
    nms_iterations = 0

    heatmap_path = os.path.join(args.save_path, "heatmaps")
    os.makedirs(heatmap_path, exist_ok=True)

    with torch.inference_mode():
        print("Warming up network...")
        for _ in range(args.warmup_iterations):
            e_example = torch.rand([1, config["event_input_channels"]] + cropped_shape).to(device)
            f_example = torch.rand([1, config["frame_input_channels"]] + cropped_shape).to(device)

            if compiled:
                e_example = e_example.to(torch.float16)
                f_example = f_example.to(torch.float16)
            # Forward pass
            if config["train_type"] == "fusion":
                input_example = (f_example, e_example)
            if config["train_type"] in ["frame", "superpoint_bn"]:
                input_example = (f_example, )
            if config["train_type"] == "event":
                input_example = (e_example, )
            if config["train_type"] == "aliked":
                input_example = (f_example.repeat(1, 3, 1, 1), )
                
            model(*input_example)

        print("Done. Starting evaluation with profiling.")

        for index, frame_timestamp in enumerate(tqdm(frame_timestamps)):
            # if index == 100: ###debug
            #     break

            prev_event_idx = current_event_idx
            frame = torch.from_numpy(frames[index]).to(torch.float32)

            # Feed events in batches
            if args.dataset_name in ["ecd", "dvs"]:
                current_event_idx = torch.argmax(torch.clamp(events_t, max=frame_timestamp))
                event_batch = events[prev_event_idx:current_event_idx]
                event_batch[:, 0] = event_batch[:, 0] - events_t[0]  

            elif args.dataset_name == "eds": 
                pass

                # current_event_idx = np.argmax(events_t > frame_timestamp * 1e6)
                # event_batch = torch.from_numpy(np.vstack([(events_t[prev_event_idx:current_event_idx] - events_t[0]) * 1e-6,
                #                                            events['x'][prev_event_idx:current_event_idx],
                #                                            events['y'][prev_event_idx:current_event_idx],
                #                                            events['p'][prev_event_idx:current_event_idx]]).T).to(device)
            elif args.dataset_name == "mvsec":
                current_event_idx = torch.argmax(torch.clamp(events_t, max=frame_timestamp))
                event_batch = events[prev_event_idx:current_event_idx][:,[2,0,1,3]]  # [t,x,y,p]
                event_batch[:, 3] = event_batch[:, 3].clamp(min=0, max=1) 
                event_batch[:, 0] = event_batch[:, 0] - events_t[0]  
                
            if len(event_batch) > 0:
                ts_gen.batch_update(event_batch)

            # Skip if not in experiment time range
            if frame_timestamp < start_time:
                continue
            if frame_timestamp > end_time:
                break

            # Experiment
            # ----------
            # Get time surface
            ts = ts_gen.get_ts()

            # For ablation study: convert to other representations
            if config["input_representation"] == "ts":
                ts = torch.max(ts[..., [2, 7]], dim=-1)[0][..., None]
            elif config["input_representation"] == "mcts_1":
                ts = ts[..., [2, 7]]
            elif config["input_representation"] == "tencode":
                ts = ts.cpu().numpy()
                ts = create_tencode_from_ts(ts[..., 4], ts[..., 9], 0.01)  # EventPoint paper reports delta_t=10ms in the experiment section
                ts = torch.from_numpy(ts).float().to(device)
                ts = ts.permute(1, 2, 0)  # channels last to keep code cleaner (tencode only used for ablation)

            ts = ts.permute(2, 0, 1).unsqueeze(0)  # channels first
            ts = ts[..., crop_mask].reshape(list(ts.shape[:-2]) + cropped_shape)

            frame = frame.unsqueeze(0).unsqueeze(0).to(device)
            frame = frame[..., crop_mask].reshape(list(frame.shape[:-2]) + cropped_shape)

            frame_img = frame[0, 0].cpu().numpy().astype(np.uint8)

            if compiled:
                ts = ts.to(torch.float16)
                frame = frame.to(torch.float16)
            start_t_measurement = time.time()


            # ####test, corrupt event ts
            # # print("corrupting event ts for testing...")
            # ts[:,4] = torch.zeros_like(ts[:,3])
            # ts[:,9] = torch.zeros_like(ts[:,8])
            # ts[:,4] = ts[:,3]
            # ts[:,9] = ts[:,8]

            # Forward pass
            if config["train_type"] == "fusion":
                input_example = (frame/255, ts)
            if config["train_type"] in ["frame", "superpoint_bn"]:
                input_example = (frame/255, )
            if config["train_type"] == "event":
                input_example = (ts, )
            if config["train_type"] == "aliked":
                # 将 frame (B, C, H, W) 的 C=1 重复为 C=3，并除以 255 归一化
                input_example = ((frame/255).repeat(1, 3, 1, 1), )

            pred = model(*input_example)

            if device == "cuda":
                torch.cuda.synchronize()
            model_inference_time += (time.time() - start_t_measurement)
            model_inference_iterations += 1

            if config["train_type"] != "aliked":
                # Non-maximum-surpression
                if compiled:
                    pred = {"prob": pred[0], "descriptors": pred[1].to(torch.float32)}
                top_k = np.max(ts.shape) // 2
                start_t_measurement = time.time()
                # kpts, _ = fast_nms(pred["prob"], config)
                kpts, _ = fast_nms(pred["prob"], config, top_k=top_k)
                if device == "cuda":
                    torch.cuda.synchronize()
                nms_time += (time.time() - start_t_measurement)
                nms_iterations += 1

                # Extract descriptors
                desc = pred["descriptors"][0, :, kpts[0][:, 0], kpts[0][:, 1]].permute(1, 0).cpu().detach().numpy()
                kpts = kpts[0].cpu().detach().numpy()
            else:
                kpts = pred['keypoints'][0]
                _, _, h, w = input_example[0].shape
                wh = torch.tensor([w - 1, h - 1],device=kpts.device)
                kpts = wh*(kpts+1)/2
                kpts = kpts.cpu().numpy() # N 2
                kpts = kpts[:, [1, 0]] # (x,y) -> (y,x)
                desc =  pred['descriptors'][0].cpu().numpy()  # N D
                nms_iterations += 1
            
            # 备份一份裁剪坐标系下的关键点用于可视化
            kpts_crop = kpts.copy() 

            # Move kpts to original positions
            kpts[:, 0] += math.ceil(crop[0] / 2)
            kpts[:, 1] += math.ceil(crop[1] / 2)

            # Get groundtruth pose
            pose = pose_interpolator.interpolate_colmap(frame_timestamp)

            pred_dict[index] = {
                "t": frame_timestamp, 
                "kpts": kpts,           # 原始坐标 (用于计算 Pose Error)
                "kpts_crop": kpts_crop, # 裁剪坐标 (用于可视化，新增这个字段)
                "desc": desc, 
                "gt_t": pose[0:3], 
                "gt_rot": pose[3:7]
            }

            # [修改 3]：存入 vis_dict，使用原始 index 作为 Key
            if args.visualize:
                # ts_img = cv2.cvtColor(visualization.ts2image(ts[0].detach().cpu().numpy().transpose(1,2,0)), cv2.COLOR_BGR2GRAY)
                ts_img = visualization.ts2image(ts[0].detach().cpu().numpy().transpose(1,2,0))
                ts_vis_dict[index] = ts_img
                frame_vis_dict[index] = frame_img

                # image_heatmap_input = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)
                # if config["train_type"] == "fusion":
                #     overlaid_img,_ = visualization.visualize_and_overlay_heatmap(image_heatmap_input, pred["features"][0], alpha=0.5)
                #     overlaid_f_img0,_ = visualization.visualize_and_overlay_heatmap(image_heatmap_input, pred["f_features"][0], alpha=0.5)
                #     overlaid_e_img0,_ = visualization.visualize_and_overlay_heatmap(image_heatmap_input, pred["e_features"][0], alpha=0.5)
                #     cv2.imwrite(os.path.join(heatmap_path ,f"step_{index}_overlay_fusion_f.png"), overlaid_f_img0)
                #     cv2.imwrite(os.path.join(heatmap_path ,f"step_{index}_overlay_fusion_e.png"), overlaid_e_img0)
                # elif config["train_type"] in ["frame", "superpoint_bn"]:
                #     overlaid_img,_ = visualization.visualize_and_overlay_heatmap(image_heatmap_input, pred["features"][-1][0], alpha=0.5)
                # elif config["train_type"] == "event":
                #     overlaid_img,_ = visualization.visualize_and_overlay_heatmap(image_heatmap_input, pred["features"][0], alpha=0.5)
                # cv2.imwrite(os.path.join(heatmap_path ,f"step_{index}_overlay.png"), overlaid_img)

    return pred_dict, ts_vis_dict, frame_vis_dict, model_inference_time, model_inference_iterations, nms_time, nms_iterations

# def process_tracked_keypoints(kpts, num_keypoints_per_step, poses, start_time, end_time, max_eval_delta_t):
#     pred_list = []
#     for pose in tqdm(poses):
#         # Skip if not in experiment time range
#         if pose[0] < start_time:
#             continue
#         if pose[0] > end_time:
#             break

#         # Find relevant keypoints (that were last tracked)
#         t = pose[0]
#         kpts_step = kpts[kpts[:, 1] <= t][::-1]
#         if kpts_step[0, 1] < t - max_eval_delta_t:  # Tracking failure (newest track is too long ago)
#             pred_list.append({"t": pose[0], "kpts": np.array([[], []]).T,
#                               "ids": [], "gt_t": pose[1:4], "gt_rot": pose[4:8]})
#             continue
#         _, idx = np.unique(kpts_step[:, 0], return_index=True)
#         kpts_step = kpts_step[np.sort(idx)[:num_keypoints_per_step]]

#         # Save keypoints and descriptors in list
#         pred_list.append({"t": pose[0], "kpts": np.stack([kpts_step[:, 3], kpts_step[:, 2]], axis=1),  # Keypoints are transposed
#                           "ids": kpts_step[:, 0], "gt_t": pose[1:4], "gt_rot": pose[4:8]})
        
#     return pred_list



def process_tracked_keypoints(args, ts_shape, kpts, num_keypoints_per_step,\
                              events, image_data, start_time, end_time, max_eval_delta_t, cropped_shape = None):
    current_event_idx = 0
    pred_dict = {}
    ts_vis_dict = {}
    frame_vis_dict = {}
    frame_timestamps = image_data["image_ts"]
    pose_interpolator = image_data["pose_interpolator"]
    frames = image_data["image"]

    #visualization
    if args.dataset_name in ["ecd", "dvs"]:
        events = torch.from_numpy(events).to(device)
        events_t = events[:, 0].clone()
    elif args.dataset_name == "eds":
        # Load timestamps into numpy array to speed-up loop
        events_t = np.array(events['t'], dtype=np.int64)
    elif args.dataset_name == "mvsec":
        events = torch.from_numpy(events[:]).to(device)
        events_t = events[:, 2].clone()


    if cropped_shape == None:
        cropped_shape = ts_shape
    settings = {"shape": ts_shape, "delta_t": args.model_delta_t}
    ts_gen = TsGenerator(settings=settings, device=device)

    for index, t in enumerate(tqdm(frame_timestamps)):
        # Skip if not in experiment time range
        if t < start_time:
            continue
        if t > end_time:
            break

        # Find relevant keypoints (that were last tracked)
        kpts_step = kpts[kpts[:, 1] <= t]
        
        if len(kpts_step) == 0:
             # 如果没有关键点，仅保存 GT
             pose_vec = pose_interpolator.interpolate_colmap(t)
             pred_dict[index] = {"t": t, "kpts": np.array([[], []]).T,
                              "ids": [], "gt_t": pose_vec[0:3], "gt_rot": pose_vec[3:7]}
             continue

        kpts_step = kpts_step[::-1] # 反转，取最新的
        
        if kpts_step[0, 1] < t - max_eval_delta_t:  # Tracking failure (too old)
            pose_vec = pose_interpolator.interpolate_colmap(t)
            pred_dict[index] = {"t": t, "kpts": np.array([[], []]).T,
                              "ids": [], "gt_t": pose_vec[0:3], "gt_rot": pose_vec[3:7]}
            continue
            
        _, idx = np.unique(kpts_step[:, 0], return_index=True)
        # 排序以保持选择顺序（或者根据置信度，这里简单取最新）
        kpts_step = kpts_step[np.sort(idx)[:num_keypoints_per_step]]

        # 获取 GT Pose (使用 interpolator 对应 Colmap 格式)
        pose_vec = pose_interpolator.interpolate_colmap(t)

        # 保存结果到字典
        pred_dict[index] = {
            "t": t, 
            "kpts": np.stack([kpts_step[:, 3], kpts_step[:, 2]], axis=1), 
            "kpts_crop": np.stack([kpts_step[:, 3], kpts_step[:, 2]], axis=1), # 填充用于可视化
            "ids": kpts_step[:, 0], 
            "gt_t": pose_vec[0:3], 
            "gt_rot": pose_vec[3:7]
        }

        prev_event_idx = current_event_idx
        frame = torch.from_numpy(frames[index]).to(torch.float32)

        # Feed events in batches
        if args.dataset_name in ["ecd", "dvs"]:
            current_event_idx = torch.argmax(torch.clamp(events_t, max=t))
            event_batch = events[prev_event_idx:current_event_idx]
            event_batch[:, 0] = event_batch[:, 0] - events_t[0]  

        elif args.dataset_name == "mvsec":
            current_event_idx = torch.argmax(torch.clamp(events_t, max=t))
            event_batch = events[prev_event_idx:current_event_idx][:,[2,0,1,3]]  # [t,x,y,p]
            event_batch[:, 3] = event_batch[:, 3].clamp(min=0, max=1) 
            event_batch[:, 0] = event_batch[:, 0] - events_t[0]  
            
        if len(event_batch) > 0:
            ts_gen.batch_update(event_batch)

        # Experiment
        # ----------
        # Get time surface
        ts = ts_gen.get_ts()
        ts = ts.permute(2, 0, 1).unsqueeze(0)  # channels first
        ts = ts[..., crop_mask].reshape(list(ts.shape[:-2]) + cropped_shape)

        frame = frame.unsqueeze(0).unsqueeze(0).to(device)
        frame = frame[..., crop_mask].reshape(list(frame.shape[:-2]) + cropped_shape)

        frame_img = frame[0, 0].cpu().numpy().astype(np.uint8)


        if args.visualize:
            # ts_img = cv2.cvtColor(visualization.ts2image(ts[0].detach().cpu().numpy().transpose(1,2,0)), cv2.COLOR_BGR2GRAY)
            ts_img = visualization.ts2image(ts[0].detach().cpu().numpy().transpose(1,2,0))
            ts_vis_dict[index] = ts_img
            frame_vis_dict[index] = frame_img
                           
    return pred_dict, ts_vis_dict, frame_vis_dict

def calculate_pose_estimation_error_pairs(pred_dict, deg_intervals, max_delta_t, pair_indices,
                                          ts_vis_dict=None, frame_vis_dict=None,
                                          ransac_threshold=3.0, img_shape = [240, 320],
                                          device = "cuda", visualize=False,
                                          save_path="results", dataset_name="ecd", is_planar=True): # <--- 新增参数
    
    # lowe_ratio_test = False
    lowe_ratio_test = True
    lowe_ratio = 0.9 #0.9
    
    if not lowe_ratio_test:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    results = {"r_error": [], "inlier_ratio": [], "num_matches":[],"num_inliers":[],"gt_rot": [], "dt": [], "reproj_error": []}
    sorted_indices = np.argsort(pair_indices[:, 0])
    pair_indices = pair_indices[sorted_indices]

    # ================== [Phase 1] 独立坐标系校准 ==================
    R_align = None
    init_bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    if dataset_name in ["ecd","dvs"]:
        # print("\n" + "="*50)
        # print("Starting Independent Coordinate System Calibration...")
        
        calibration_cache = []
        calibration_pairs_count = 0
        
        # 预扫描前 50 个 pair 寻找高质量匹配用于校准
        # 注意：这里只做匹配和计算R，不记录结果，纯为了找 R_align
        for pair_idx in range(min(len(pair_indices), 100)):
            if len(calibration_cache) >= 50: # 收集够15个样本就停止
                break
                
            idx0 = pair_indices[pair_idx][0]    
            idx1 = pair_indices[pair_idx][1]
            if idx0 not in pred_dict or idx1 not in pred_dict: continue
            data0, data1 = pred_dict[idx0], pred_dict[idx1]

                        # 1. 快速匹配 (BFMatcher 或 Track IDs)
            matched_kpts = None
            matches_len = 0 # 用于后续内点率计算

            if "ids" in data0:
                # 使用 Track IDs 匹配
                _, idxs_step, idxs_step_dist = np.intersect1d(data0["ids"], data1["ids"], return_indices=True)
                if len(idxs_step) < 10: continue
                # 注意：matched_kpts 的格式需要是 (N, 2, 2)，最后一维是 [y, x]
                # data0["kpts"] 本身就是 [y, x]
                matched_kpts = np.array([data0["kpts"][idxs_step], data1["kpts"][idxs_step_dist]]).transpose(1, 0, 2)
                matches_len = len(matched_kpts)
            
            elif "desc" in data0:
                # 使用描述子匹配
                matches = init_bf.match(data0["desc"], data1["desc"])
                if len(matches) < 10: continue
                matched_kpts = np.array([[data0["kpts"][m.queryIdx], data1["kpts"][m.trainIdx]] for m in matches]).astype(np.float32)
                matches_len = len(matches)
            
            else:
                continue

            # 2. 快速解算 Pose
            matched_kpts = np.roll(matched_kpts, 1, axis=-1) # (y,x) -> (x,y)
            pts1 = cv2.undistortPoints(matched_kpts[:, 0].reshape(-1, 1, 2), camera_matrix, distortion_coeffs, P=camera_matrix)
            pts2 = cv2.undistortPoints(matched_kpts[:, 1].reshape(-1, 1, 2), camera_matrix, distortion_coeffs, P=camera_matrix)
            
            try:
                E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=ransac_threshold)
                if E is None: continue
                _, est_R, _, _ = cv2.recoverPose(E, matched_kpts[:, 0], matched_kpts[:, 1], camera_matrix)
            except: continue

            # 3. 严格筛选校准样本
            # 条件：内点率高 (>0.5) 且 旋转幅度大 (>2度)
            # 这样保证用于校准的 "Est" 绝对是准确的
            inlier_ratio = np.sum(mask) / matches_len
            est_rot = R.from_matrix(est_R)
            est_mag = np.linalg.norm(est_rot.as_rotvec(degrees=True))
            
            gt_rot_raw = R.from_quat(data0["gt_rot"]).inv() * R.from_quat(data1["gt_rot"])
            gt_mag = np.linalg.norm(gt_rot_raw.as_rotvec(degrees=True))

            if inlier_ratio > 0.5 and est_mag > 2.0 and gt_mag > 2.0:
                calibration_cache.append((est_rot, gt_rot_raw))
        
        # 4. 计算校准矩阵
        if len(calibration_cache) > 5:
            axes = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]),
                    np.array([-1,0,0]), np.array([0,-1,0]), np.array([0,0,-1])]
            best_score = float('inf')
            
            for x_axis, y_axis in itertools.permutations(axes, 2):
                if np.dot(x_axis, y_axis) == 0:
                    z_axis = np.cross(x_axis, y_axis)
                    R_cand = R.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))
                    
                    curr_error = 0
                    for e_r, g_r in calibration_cache:
                        g_r_cam = R_cand * g_r * R_cand.inv()
                        curr_error += np.linalg.norm((e_r * g_r_cam.inv()).as_rotvec())
                    
                    if curr_error < best_score:
                        best_score = curr_error
                        R_align = R_cand
            
            # print(f"Calibration Successful using {len(calibration_cache)} pairs.")
            # print(f"Align Matrix (Body -> Camera):\n{R_align.as_matrix()}")
            
            # # 打印直观的轴映射关系
            # m = R_align.as_matrix()
            # axis_names = ['X', 'Y', 'Z']
            # print("Mapping visualization:")
            # for i in range(3): # Est axes
            #     idx = np.argmax(np.abs(m[i, :]))
            #     sign = '+' if m[i, idx] > 0 else '-'
            #     print(f"  Camera {axis_names[i]} <== {sign} Body {axis_names[idx]}")
        else:
            print("Calibration Failed: Not enough high-quality rotating frames found. Using Identity.")

        # print("="*50 + "\n")
    # ==============================================================
    if dataset_name == "mvsec": 
        # manual_align_matrix = np.array([[ 1.,  0.,  0.],
        #                                 [ 0.,  0., -1.],
        #                                 [ 0.,  1.,  0.]])
        # R_align = R.from_matrix(manual_align_matrix)

        # R_cam_lidar = np.array([
        #     [0, -1,  0],
        #     [ 0, 0, -1],
        #     [ 1, 0,  0]
        # ])

        # R_fix = np.array([
        #     [0, 1, 0],
        #     [0, 0, 1],
        #     [1, 0, 0]
        # ])
        
        # manual_align_matrix = R_fix @ R_cam_lidar
        # R_align = R.from_matrix(manual_align_matrix)

        # manual_align_matrix = np.array([[ 0.,  -1.,  0.],
        #                         [ 0.,  0., -1.],
        #                         [ 1.,  0.,  0.]])
        # R_align = R.from_matrix(manual_align_matrix)
        R_align = None # 先不做校准，看看结果如何
    
    for pair in tqdm(range(len(pair_indices))):

        idx0 = pair_indices[pair][0]    
        idx1 = pair_indices[pair][1]
        
        if idx0 not in pred_dict or idx1 not in pred_dict:
            continue

        data0 = pred_dict[idx0]
        data1 = pred_dict[idx1]

        # Calculate groundtruth rotation change
        gt_rot_raw = R.from_quat(data0["gt_rot"]).inv() * R.from_quat(data1["gt_rot"])
        
        if R_align is not None:
            gt_rot = R_align * gt_rot_raw * R_align.inv()
        else:
            gt_rot = gt_rot_raw
        
        abs_gt_rot_deg = np.linalg.norm(gt_rot.as_rotvec(degrees=True))

        # Match
        matches = [] # 初始化空列表，防止后面引用报错
        if "ids" in data0.keys():
            # 利用 numpy 的 intersect1d 找到共有 ID 的索引
            # idxs_step: data0 中共有 ID 的索引
            # idxs_step_dist: data1 中共有 ID 的索引
            _, idxs_step, idxs_step_dist = np.intersect1d(data0["ids"], data1["ids"], return_indices=True)
            
            # 提取匹配点对
            # 原数据格式为 (N, 2)，堆叠后变为 (2, N, 2)，再 transpose 为 (N, 2, 2)
            # data0["kpts"] 已经是 [y, x] 格式（或者根据 preprocess 确定），这里直接提取
            matched_kpts = np.array([data0["kpts"][idxs_step], data1["kpts"][idxs_step_dist]]).transpose(1, 0, 2)

            # 如果需要可视化，构建假的 DMatch 对象
            if visualize: # 或者直接用 if True，反正后面只会因 visualize 参数决定画不画
                matches = [cv2.DMatch(_queryIdx=idxs_step[i], _trainIdx=idxs_step_dist[i], _imgIdx=0, _distance=0.0) for i in range(len(matched_kpts))]

        elif "desc" in data0.keys():
            if lowe_ratio_test:
                # KNN match
                knn_matches = bf.knnMatch(data0["desc"], data1["desc"], k=2)
                # Apply Lowe's ratio test
                good_matches = []
                for m, n in knn_matches:
                    if m.distance < lowe_ratio * n.distance:
                        good_matches.append(m)
                matches = good_matches
            else:
                matches = bf.match(data0["desc"], data1["desc"])

            matched_kpts = np.array([[data0["kpts"][m.queryIdx], data1["kpts"][m.trainIdx]] for m in matches]).astype(np.float32)

        else:
            # 如果既没有 ids 也没有 desc，说明该帧无效
            matched_kpts = np.zeros((0, 2, 2), dtype=np.float32)

        # ================= 1. 数据准备 =================
        matched_kpts = np.roll(matched_kpts, 1, axis=-1)
        if len(matched_kpts) > 8:
            pts1 = matched_kpts[:, 0].reshape(-1, 1, 2).astype(np.float64)
            pts2 = matched_kpts[:, 1].reshape(-1, 1, 2).astype(np.float64)

            # 显式去畸变
            pts1_undist = cv2.undistortPoints(pts1, camera_matrix, distortion_coeffs, P=camera_matrix)
            pts2_undist = cv2.undistortPoints(pts2, camera_matrix, distortion_coeffs, P=camera_matrix)
        else:
            # 点太少，直接跳过
            results["r_error"].append(180.0)
            results["inlier_ratio"].append(0.0)
            results["num_matches"].append(len(matched_kpts))
            results["num_inliers"].append(0)
            results["gt_rot"].append(abs_gt_rot_deg)
            delta_t = abs(data1["t"] - data0["t"])
            results["dt"].append(delta_t)
            results["reproj_error"].append(100.0)
            # print(f"Warning: Not enough matches ({len(matched_kpts)}) for pair {idx0} to {idx1}. Skipping pose estimation.")
            continue

        # 初始化结果
        reproj_err = 0.0
        error_angle_deg = 0.0 # 默认为0，或者设为nan
        num_inliers = 0
        
        # 辅助函数：计算两个旋转矩阵的角度差 (deg)
        def calc_rot_error(R_est, R_gt_obj):
            try:
                est_rot_obj = R.from_matrix(R_est)
                rot_diff = est_rot_obj * R_gt_obj.inv()
                return np.linalg.norm(rot_diff.as_rotvec(degrees=True))
            except Exception:
                return 180.0

        # ================= 2. 分支 A: 平面场景 (Poster/Shapes) -> H =================
        if is_planar:
            try:
                H, mask_h = cv2.findHomography(
                    pts1_undist, pts2_undist, cv2.RANSAC, ransacReprojThreshold=3.0
                )
                
                if H is not None:
                    num_inliers = np.sum(mask_h)
                    
                    # --- [Part 1] 计算角度误差 (从 H 分解) ---
                    if num_inliers > 10: # 至少要有一定数量的点才能做分解
                        # 分解 H 得到 4 组可能的解 (包含 R 和 t)
                        num_sol, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, camera_matrix)
                        
                        # 遍历 4 个解，找到和 GT 最接近的那个 (Oracle Evaluation)
                        min_angle_error = 180.0
                        for i in range(num_sol):
                            current_R = Rs[i]
                            err = calc_rot_error(current_R, gt_rot)
                            if err < min_angle_error:
                                min_angle_error = err
                        error_angle_deg = min_angle_error
                    else:
                        error_angle_deg = 180.0

                    # --- [Part 2] 计算重投影误差 (Symmetric Transfer Error) ---
                    mask_bool = mask_h.ravel().astype(bool)
                    if np.sum(mask_bool) > 0:
                        p1_in = pts1_undist[mask_bool].reshape(-1, 2)
                        p2_in = pts2_undist[mask_bool].reshape(-1, 2)
                        
                        # Forward
                        ones = np.ones((len(p1_in), 1))
                        p1_h = np.hstack([p1_in, ones])
                        p2_pred_h = (H @ p1_h.T).T
                        p2_pred = p2_pred_h[:, :2] / (p2_pred_h[:, 2:] + 1e-8)
                        err_fwd = np.linalg.norm(p2_in - p2_pred, axis=1)
                        
                        # Backward
                        H_inv = np.linalg.inv(H)
                        p2_h = np.hstack([p2_in, ones])
                        p1_pred_h = (H_inv @ p2_h.T).T
                        p1_pred = p1_pred_h[:, :2] / (p1_pred_h[:, 2:] + 1e-8)
                        err_bwd = np.linalg.norm(p1_in - p1_pred, axis=1)
                        
                        reproj_err = np.mean((err_fwd + err_bwd) / 2.0)
                else:
                    num_inliers = 0
                    reproj_err = 100.0
                    error_angle_deg = 180.0

            except cv2.error:
                num_inliers = 0
                reproj_err = 100.0
                error_angle_deg = 180.0

        # ================= 3. 分支 B: 3D 场景 (Boxes) -> E =================
        else:
            try:
                E, mask_e = cv2.findEssentialMat(
                    pts1_undist, pts2_undist, camera_matrix, 
                    method=cv2.RANSAC, prob=0.999, threshold=ransac_threshold
                )
                
                if E is not None and E.shape == (3, 3):
                    # 从 E 恢复唯一的 R, t (Cheirality check included)
                    inliers_count, est_R, est_t, mask_pose = cv2.recoverPose(
                        E, pts1_undist, pts2_undist, camera_matrix
                    )
                    num_inliers = inliers_count # 这里的 inlier 是通过了两个校验的

                    # --- [Part 1] 计算角度误差 ---
                    error_angle_deg = calc_rot_error(est_R, gt_rot)

                    # --- [Part 2] 计算重投影误差 (Sampson Error) ---
                    # 注意：recoverPose 返回的 mask_pose 可能和 findEssentialMat 的 mask_e 不同
                    # 我们尽量用通过了 recoverPose 检验的 mask
                    if mask_pose is not None and np.sum(mask_pose) > 0:
                        mask_bool = mask_pose.ravel().astype(bool)
                        
                        p1_in = pts1_undist[mask_bool].reshape(-1, 2)
                        p2_in = pts2_undist[mask_bool].reshape(-1, 2)

                        def skew_symmetric(v):
                            return np.array([[0, -v[2], v[1]],
                                                [v[2], 0, -v[0]],
                                                [-v[1], v[0], 0]])

                        # 计算 F 矩阵: F = K^-T * E * K^-1
                        # E = [t]x * R
                        K_inv = np.linalg.inv(camera_matrix)
                        t_vec = est_t.flatten()
                        E_mat = skew_symmetric(t_vec) @ est_R
                        F_mat = K_inv.T @ E_mat @ K_inv
                        
                        # 构造齐次坐标
                        p1_h = np.hstack([p1_in, np.ones((len(p1_in), 1))])
                        p2_h = np.hstack([p2_in, np.ones((len(p2_in), 1))])
                        
                        # l2 = E * p1, l1 = E.T * p2
                        lines2 = (F_mat @ p1_h.T).T 
                        lines1 = (F_mat.T @ p2_h.T).T
                        
                        # Sampson distance
                        num = np.abs(np.sum(p2_h * lines2, axis=1))
                        den1 = np.sqrt(lines1[:,0]**2 + lines1[:,1]**2)
                        den2 = np.sqrt(lines2[:,0]**2 + lines2[:,1]**2)
                        reproj_err = np.mean(num / (den1 + den2 + 1e-8) * 2.0)
                    else:
                        reproj_err = 100.0

                else:
                    num_inliers = 0
                    reproj_err = 100.0
                    error_angle_deg = 180.0

            except cv2.error:
                num_inliers = 0
                reproj_err = 100.0
                error_angle_deg = 180.0

        if num_inliers == 0:
            print(f"Warning: Pose estimation failed for pair {idx0} to {idx1}.")
            pass 

        results["r_error"].append(error_angle_deg)
        results["inlier_ratio"].append(num_inliers / max(len(matched_kpts), 1))
        results["num_matches"].append(len(matched_kpts))
        results["num_inliers"].append(num_inliers)
        results["gt_rot"].append(abs_gt_rot_deg)
        results["dt"].append(data1["t"] - data0["t"])
        results["reproj_error"].append(reproj_err) # [新增]

        # [可视化部分修改]
        if visualize and ts_vis_dict is not None:
            if idx0 in ts_vis_dict and idx1 in ts_vis_dict:
                # 1. 原有的匹配可视化
                kpts_cv2_0 = [cv2.KeyPoint(float(data0["kpts_crop"][i, 1]), float(data0["kpts_crop"][i, 0]), 1) for i in range(len(data0["kpts"]))]
                kpts_cv2_1 = [cv2.KeyPoint(float(data1["kpts_crop"][i, 1]), float(data1["kpts_crop"][i, 0]), 1) for i in range(len(data1["kpts"]))]

                matched_img_e = cv2.drawMatches(ts_vis_dict[idx0], kpts_cv2_0, ts_vis_dict[idx1], kpts_cv2_1, matches,
                                            None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
                matched_img_f = cv2.drawMatches(frame_vis_dict[idx0], kpts_cv2_0, frame_vis_dict[idx1], kpts_cv2_1, matches,
                                            None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
                
                matched_img = np.concatenate((matched_img_e, matched_img_f), axis=0)
                cv2.imwrite(os.path.join(save_path, f"matches_step_{idx0}_to_{idx1}_gtrot_{int(abs_gt_rot_deg)}_deg.png"), matched_img)

                # # ================== 新增：去畸变效果检查 ==================
                # # 只保存前 20 张用于检查，避免生成太多图片
                # if pair < 20:
                #     img_raw = frame_vis_dict[idx0]
                #     # 使用全局内参进行图像去畸变
                #     img_undist = cv2.undistort(img_raw, camera_matrix, distortion_coeffs)
                    
                #     # 拼接图片：左边是原图(Distorted)，右边是去畸变(Undistorted)
                #     vis_undist_compare = np.concatenate((img_raw, img_undist), axis=1)
                    
                #     # 保存对比图
                #     cv2.imwrite(os.path.join(save_path, f"undist_check_{idx0}.png"), vis_undist_compare)
                # # ========================================================
        # [修改 6]: 可视化部分改用字典访问 (已修改为显示去畸变后的结果)
        if visualize and ts_vis_dict is not None:
            # 确保字典里有图（虽然 idx check 过了，但双重保险）
            if idx0 in ts_vis_dict and idx1 in ts_vis_dict:
                # --- 1. 准备图像和进行去畸变 ---
                # 获取原图
                img_ts0_raw = ts_vis_dict[idx0]
                img_ts1_raw = ts_vis_dict[idx1]
                img_frame0_raw = frame_vis_dict[idx0]
                img_frame1_raw = frame_vis_dict[idx1]

                # 对图像进行去畸变
                # 注意：这会使用全局的 camera_matrix 和 distortion_coeffs
                img_ts0_undist = cv2.undistort(img_ts0_raw, camera_matrix, distortion_coeffs)
                img_ts1_undist = cv2.undistort(img_ts1_raw, camera_matrix, distortion_coeffs)
                img_frame0_undist = cv2.undistort(img_frame0_raw, camera_matrix, distortion_coeffs)
                img_frame1_undist = cv2.undistort(img_frame1_raw, camera_matrix, distortion_coeffs)

                # --- 2. 准备去畸变后的关键点 ---
                # 利用上面已经计算好的 pts1_undist 和 pts2_undist
                # 它们的形状是 (N, 1, 2), 坐标已经是 (x, y) 格式，且 P=camera_matrix 保证了它们在像素坐标系下
                # 将其转换为 cv2.KeyPoint 对象用于绘图
                kpts_undist_0 = [cv2.KeyPoint(float(pt[0][0]), float(pt[0][1]), 1) for pt in pts1_undist]
                kpts_undist_1 = [cv2.KeyPoint(float(pt[0][0]), float(pt[0][1]), 1) for pt in pts2_undist]

                # ... (前文代码不变: 去畸变图像、生成 kpts_undist_0 等) ...
                
                # --- 3. [关键修改] 重建匹配关系 ---
                # 因为 kpts_undist 已经是筛选过且一一对应的点（第i个对应第i个）
                # 所以原来的 'matches' 里的索引(指向原始大数据集)已经失效了
                # 我们需要创建一个新的匹配列表，让它指向这个只有 N 个点的新列表
                num_points = len(kpts_undist_0)
                new_matches = []
                for i in range(num_points):
                    # 创建一个简单的 1对1 匹配: 第 i 个点匹配第 i 个点
                    new_matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0))

                # --- 4. 绘制匹配图 ---
                # 注意：这里传入的是 new_matches
                
                # 绘制 Time Surface 匹配
                matched_img_e = cv2.drawMatches(img_ts0_undist, kpts_undist_0, img_ts1_undist, kpts_undist_1, new_matches,
                                            None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
                # 绘制 Frame 匹配
                matched_img_f = cv2.drawMatches(img_frame0_undist, kpts_undist_0, img_frame1_undist, kpts_undist_1, new_matches,
                                            None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
                
                matched_img = np.concatenate((matched_img_e, matched_img_f), axis=0)
                
                cv2.imwrite(os.path.join(save_path, f"undistorted_matches_step_{idx0}_to_{idx1}_gtrot_{int(abs_gt_rot_deg)}_deg.png"), matched_img)
    return results



def calculate_auc(results, auc_threshold_list):
    # Create precision-recall curve
    sorted_idxs = np.argsort(results["r_error"])
    results["r_error"] = [0] + np.array(results["r_error"])[sorted_idxs].tolist()
    results["inlier_ratio"] = [0] + np.array(results["inlier_ratio"])[sorted_idxs].tolist()
    results["gt_rot"] = [0] + np.array(results["gt_rot"])[sorted_idxs].tolist()

    # Adapted from LoFTR (Apache-2.0); see THIRD_PARTY_LICENSES.txt:
    # https://github.com/zju3dv/LoFTR/blob/df7ca80f917334b94cfbe32cc2901e09a80e70a8/src/utils/metrics.py#L148C2-L154C42
    aucs = []
    recall = list(np.linspace(0, 1, len(results["r_error"])))
    for thr in auc_threshold_list:
        last_index = np.searchsorted(results["r_error"], thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = results["r_error"][:last_index] + [thr]
        aucs.append(np.trapezoid(y, x) / thr)
    print(f"AUCs for {auc_threshold_list}: {aucs}.")

    return aucs

def generate_pair_from_sequence(
    image_data: Dict[str, Any], time_window: int, method: str, num_pairs: int
):

    image_length = len(image_data["image"])
    print(image_length)
    random_indices1 = np.random.randint(0, image_length - 1, num_pairs)
    random_indices2 = []
    for i in random_indices1:
        if method == "uniform":
            index2 = np.random.randint(i, min(image_length - 1, i + time_window))
        random_indices2.append(index2)

    random_indices2 = np.array(random_indices2)

    indices = np.stack([random_indices1, random_indices2], axis=1)

    windows = np.abs(random_indices1 - random_indices2)

    return indices

def generate_pair_from_sequence_EC_blur(
    image_data: Dict[str, Any], 
    time_window: int, 
    method: str, 
    num_pairs: int,
    min_rot_deg: float = 5.0  # 注意：这里阈值单位改成了【度】
    ):
    """
    筛选出旋转大于 min_rot_deg 的帧作为模糊候选，并生成匹配对。
    """
    length = image_data["length"]
    image_ts = image_data["image_ts"]
    pose_interpolator = image_data["pose_interpolator"]
    
    # --- 第一步：预筛选 (Pre-screening) ---
    blur_candidates = []
    
    # 遍历所有相邻帧 (i, i+1)
    for i in range(length - 1):
        try:
            t0 = image_ts[i]
            t1 = image_ts[i+1]
            
            # 1. 获取 4x4 位姿矩阵
            pose0 = pose_interpolator.interpolate(t0)
            pose1 = pose_interpolator.interpolate(t1)
            
            r0 = R.from_matrix(pose0[:3, :3])
            r1 = R.from_matrix(pose1[:3, :3])
            
            relative_rot = r0.inv() * r1
            rot_deg = np.linalg.norm(relative_rot.as_rotvec(degrees=True))
            
            if rot_deg > min_rot_deg:
                blur_candidates.append(i)

        except Exception as e:
            # 遇到插值错误或奇异矩阵时跳过
            continue

    if not blur_candidates:
        raise ValueError(f"未找到相邻旋转大于 {min_rot_deg}° 的样本，请调低 min_rot_deg。")
    
    print(f"筛选完成：在 {length} 帧中找到了 {len(blur_candidates)} 个满足 >{min_rot_deg}° 的模糊候选帧。")

    # --- 第二步：从候选池生成 Pair (Sampling) ---
    
    candidates_arr = np.array(blur_candidates)
    
    # 允许重复采样以满足 num_pairs 需求
    selected_indices_1 = np.random.choice(candidates_arr, size=num_pairs, replace=True)
    
    indices = []
    for idx1 in selected_indices_1:
        # 生成 idx2
        if method == "uniform":
            low = idx1 + 1
            high = min(length - 1, idx1 + time_window)
            
            if high <= low:
                idx2 = min(length - 1, idx1 + 1)
            else:
                idx2 = np.random.randint(low, high + 1)
        else:
            idx2 = min(length - 1, idx1 + 1)
            
        indices.append([idx1, idx2])

    return np.array(indices)

def generate_pair_from_sequence_EC_double_blur(
    image_data: Dict[str, Any], 
    time_window: int, 
    num_pairs: int,
    min_inst_vel_deg: float = 3.0,     # 模糊阈值：idx1 和 idx2 都必须大于此瞬时速度
    max_baseline_rot_deg: float = 20.0, # 重叠上限：idx1-idx2 总旋转不能太大
    min_baseline_rot_deg: float = 3.0   # 重叠下限：idx1-idx2 总旋转不能太小
    ):
    
    length = image_data["length"]
    image_ts = image_data["image_ts"]
    pose_interpolator = image_data["pose_interpolator"]
    
    # --- 第一步：找到所有模糊帧 (Identify ALL blurry frames) ---
    blur_candidates = []
    
    print(f"正在筛选所有模糊帧 (瞬时旋转 > {min_inst_vel_deg}°)...")
    
    # 这一步找出所有正在剧烈运动的时刻
    for i in range(length - 1):
        try:
            t_curr = image_ts[i]
            t_next = image_ts[i+1]
            
            pose_curr = pose_interpolator.interpolate(t_curr)
            pose_next = pose_interpolator.interpolate(t_next)
            
            r_curr = R.from_matrix(pose_curr[:3, :3])
            r_next = R.from_matrix(pose_next[:3, :3])
            
            # 计算瞬时角速度
            inst_rot = np.linalg.norm((r_curr.inv() * r_next).as_rotvec(degrees=True))
            
            if inst_rot > min_inst_vel_deg:
                blur_candidates.append(i)

        except Exception:
            continue

    if len(blur_candidates) < 2:
        raise ValueError(f"找到的模糊帧数量不足 ({len(blur_candidates)}个)，无法组成配对。请降低 min_inst_vel_deg。")
    
    print(f"共找到 {len(blur_candidates)} 个模糊帧。开始进行 'Blur-to-Blur' 配对搜索...")

    # --- 第二步：在模糊帧集合内部寻找有效配对 (Search within candidates) ---
    valid_pairs = []
    
    # 只需要遍历模糊帧列表，不需要遍历整个视频序列
    # i 是 blur_candidates 的索引，idx1 是实际的帧号
    for i in tqdm(range(len(blur_candidates)), desc="Searching Blur-to-Blur Pairs"):
        idx1 = blur_candidates[i]
        
        t1 = image_ts[idx1]
        pose1 = pose_interpolator.interpolate(t1)
        r1 = R.from_matrix(pose1[:3, :3])
        r1_inv = r1.inv()
        
        # 从 i+1 开始遍历剩下的模糊帧作为 idx2
        for j in range(i + 1, len(blur_candidates)):
            idx2 = blur_candidates[j]
            
            # [优化] 时间窗口检查：
            # 如果 idx2 已经超出了时间窗口，因为 list 是有序的，后面的肯定更远，直接 break
            if idx2 > idx1 + time_window:
                break
            
            try:
                t2 = image_ts[idx2]
                pose2 = pose_interpolator.interpolate(t2)
                r2 = R.from_matrix(pose2[:3, :3])
                
                # 计算相对旋转 (Baseline Rotation)
                baseline_rot = np.linalg.norm((r1_inv * r2).as_rotvec(degrees=True))
                
                # 几何约束检查：必须保证两帧虽然模糊，但还有重叠
                if min_baseline_rot_deg < baseline_rot < max_baseline_rot_deg:
                    valid_pairs.append([idx1, idx2])
            
            except Exception:
                continue

    # --- 第三步：结果处理与采样 ---
    all_valid_indices = np.array(valid_pairs)
    total_found = len(all_valid_indices)
    
    print(f"搜索完成：共找到 {total_found} 对双端模糊匹配对。")
    print(f"约束条件: 瞬时速度>{min_inst_vel_deg}°, 相对位姿 {min_baseline_rot_deg}°~{max_baseline_rot_deg}°")

    if total_found == 0:
         raise ValueError("未找到符合条件的对，可能是时间窗口太小或角度限制太严。")

    if total_found > num_pairs:
        selected_row_indices = np.random.choice(total_found, size=num_pairs, replace=False)
        final_indices = all_valid_indices[selected_row_indices]
        print(f"随机抽取 {num_pairs} 对进行测试。")
    else:
        final_indices = all_valid_indices
        print(f"返回所有 {total_found} 对。")

    return final_indices


# Fix seed for reproducibility
fix_seed()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", help="Root directory of dataset")
parser.add_argument("--dataset_name", default="", help="Dataset used for evaluation. Supported datasets are Event Camera Dataset ('ecd') and Event-aided Direct Sparse Odometry ('eds').")
parser.add_argument("--sequence_names", nargs="*", default=[], help="Names of evaluation sequences")
parser.add_argument("--config", default="config/fe_fusionpoint.yaml", help="Parameter configuration.")
parser.add_argument("--model", default="", help="Model checkpoint override; defaults to checkpoint in the model config.")
parser.add_argument("--model_delta_t", nargs="*", default=[0.001, 0.003, 0.01, 0.03, 0.1], type=float, help="Time delta of time surfaces")
# parser.add_argument("--model_delta_t", nargs="*", default=[0.0003, 0.001, 0.003, 0.01, 0.03], type=float, help="Time delta of time surfaces")
parser.add_argument("--max_eval_delta_t", nargs="*", default=2.0, type=float, help="List of durations between testing the repeatability")
parser.add_argument("--auc_thresholds",  nargs="*", default=[5.0, 10.0, 20.0], help="Thresholds for AUC evaluation.")
parser.add_argument('--visualize', default=False, action=argparse.BooleanOptionalAction, help="Visualize matches, only for debugging")
parser.add_argument("--kpts_dir", default="", help="Directory where keypoints are loaded from. If not specified (default), SuperEvent will predict the keypoints.")
parser.add_argument("--num_kpts_per_step", default=-1, help="Number of keypoints tracked in parallel. Only used when 'kpts_dir' is specified. -1 to infer (default).")
parser.add_argument('--compile', default=False, action=argparse.BooleanOptionalAction, help="torch.compile model before inference.")
parser.add_argument("--save_path", default="results", help="Path where results are saved.")
parser.add_argument("--log_file", default="", help="Evaluation log path; defaults to <save_path>/evaluation.log.")

parser.add_argument("--pairs_root", default="", help="Read fixed sequence/<sequence>_blur_<threshold>.txt files from this root.")
parser.add_argument("--calibration_root", default="", help="Directory containing calib.txt and, for MVSEC, camchain-imucam-outdoor_night.yaml.")
parser.add_argument("--require_existing_pairs", default=True, action=argparse.BooleanOptionalAction, help="Require fixed evaluation pairs; disable explicitly for custom pair generation.")
parser.add_argument("--profile", default=False, action=argparse.BooleanOptionalAction, help="Run optional thop profiling before evaluation.")

parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
parser.add_argument("--warmup_iterations", type=int, default=100)
parser.add_argument("--blur_thresholds", nargs="+", type=int, default=None)
args = parser.parse_args()

if not args.kpts_dir:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if not args.model:
        args.model = config.get("checkpoint", "")
    if not args.model:
        parser.error("Specify --model or set checkpoint in the model config.")
    if not os.path.isfile(args.model):
        parser.error(f"Model checkpoint does not exist: {args.model}")

if not args.dataset_name:
    args.dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
    print(f"Evaluating on dataset {args.dataset_name}.")

supported_datasets = ["ecd", "eds", "mvsec", "dvs"]
assert args.dataset_name in supported_datasets, f"Datset {args.dataset_name} not supported. Please use one of {supported_datasets}."

if args.dataset_name in ["ecd", "mvsec"]:
    evaluation_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")
    if not args.pairs_root:
        args.pairs_root = os.path.join(evaluation_root, "pairs", args.dataset_name)
    if not args.calibration_root:
        args.calibration_root = os.path.join(evaluation_root, "calibration", args.dataset_name)

input_root = os.path.realpath(args.dataset_path)
output_root = os.path.realpath(args.save_path)
log_path = os.path.realpath(args.log_file or os.path.join(args.save_path, "evaluation.log"))
if any(os.path.commonpath([input_root, path]) == input_root for path in [output_root, log_path]):
    parser.error("Choose output and log paths outside the input dataset.")
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(os.path.dirname(log_path), exist_ok=True)
log_file = open(log_path, "w")

if not args.sequence_names:  # Use default sequences
    if args.dataset_name == "ecd":
        # args.sequence_names = ["hdr_boxes"]
        # args.sequence_names = ["boxes_6dof", "boxes_rotation", "poster_6dof", "poster_rotation", "shapes_6dof", "shapes_rotation", "dynamic_6dof", "hdr_boxes", "hdr_poster"]
        args.sequence_names = ["boxes_6dof", "poster_6dof", "dynamic_6dof", "hdr_boxes", "hdr_poster"]
    if args.dataset_name == "dvs":
        # 定义父文件夹名称
        # dvs_parent_folders = ["bear_dataset", "boxes_dataset", "desk_dataset"]
        dvs_parent_folders = ["boxes_00_dataset", "desk_00_dataset"]
        args.sequence_names = []
        for folder in dvs_parent_folders:
            folder_path = os.path.join(args.dataset_path, folder)
            if os.path.exists(folder_path):
                # 遍历所有子目录 (排除 calib.txt 等文件)
                sub_dirs = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
                for sub in sub_dirs:
                    # if sub in ["normal_1", "normal_2", "blur_1", "blur_2", "dark_1", "dark_2", "hdr_1", "hdr_2"]:
                    if sub in ["normal_1", "normal_2", "blur_1", "blur_2", "dark_1", "dark_2"]:
                    # 组合成 "bear/blur_1" 格式
                        args.sequence_names.append(os.path.join(folder, sub))
    elif args.dataset_name == "eds":
        args.sequence_names = ["peanuts_dark", "peanuts_light", "rocket_earth_light", "rocket_earth_dark", "ziggy_and_fuzz", "ziggy_and_fuzz_hdr", "peanuts_running", "all_characters"]
    elif args.dataset_name == "mvsec":
        args.sequence_names = ["outdoor_night1", "outdoor_night2", "outdoor_night3"]

if args.dataset_name == "ecd":
    ts_shape = [180, 240]
    pose_file_name = "groundtruth.txt"
    # events_file_name = "events.txt"
    events_file_name = "events.npy"
    image_timestampes_file_name = "images.txt"
    frame_dir_name = "images"
    ransac_threshold = 1.0 #1.0
elif args.dataset_name == "dvs":
    ts_shape = [260, 346]
    pose_file_name = "groundtruth.txt"
    events_file_name = "events.npy"
    image_timestampes_file_name = "images.txt"
    frame_dir_name = "images"
    ransac_threshold = 1.0 #1.0

elif args.dataset_name == "eds":
    ts_shape = [480, 640]
    pose_file_name = "stamped_groundtruth.txt"
    events_file_name = "events.h5"
    image_timestampes_file_name = "images_timestamps.txt"
    frame_dir_name = "images"
    ransac_threshold = 3.0
elif args.dataset_name == "mvsec":
    ts_shape = [260, 346]
    ransac_threshold = 1.0 #2.0
    mvsec_data_path_list = []
    mvsec_data_path_list.append(os.path.join(args.dataset_path, seq + "_data.hdf5") for seq in args.sequence_names)
    pair_file_path_list = []
    pair_file_path_list.append(os.path.join(args.dataset_path, seq + "_final_indices.txt") for seq in args.sequence_names) 



# Time measurement
total_model_inference_time = 0.
total_model_inference_iterations = 0
total_nms_time = 0.
total_nms_iterations = 0

# 定义模糊阈值列表
if args.dataset_name == "ecd":
    blur_thresholds = [1, 3, 5, 7] 
elif args.dataset_name == "mvsec":
    blur_thresholds = [0]
else:
    blur_thresholds = [1, 3] 

if args.blur_thresholds is not None:
    blur_thresholds = args.blur_thresholds

# 用来存储所有阈值的汇总结果，方便最后打印对比表格

final_summary = {thresh: {"r_error": [], "inlier_ratio": [], "num_matches":[], "num_inliers":[], "gt_rot": [], "reproj_error": []} for thresh in blur_thresholds}
final_AUC = {thresh: {"AUC@5": [], "AUC@10": [], "AUC@20": []} for thresh in blur_thresholds}

if args.kpts_dir:
    experiment_name = "pose_estimation_" + args.dataset_name + "_" + os.path.basename(os.path.normpath(args.kpts_dir)) + "_at_time_" + time.strftime("%Y%m%d-%H%M%S")
    device = "cuda"
    crop_mask = torch.ones(ts_shape, dtype=bool)
else:
    experiment_name = "pose_estimation_" + args.dataset_name + "_" + os.path.splitext(os.path.basename(args.model))[0] + "_at_time_" + time.strftime("%Y%m%d-%H%M%S")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if args.device != "auto":
        device = args.device
    print(f"Using {device} device")

    print("Loaded config from", args.config)

    if config["train_type"] == "frame" or config["train_type"] == "event":
        config["input_channels"] = config["frame_input_channels"] if config["train_type"] == "frame" else config["event_input_channels"]
    if "backbone" in config:
        # Load backbone config and add
        backbone_config_path = os.path.join(os.path.dirname(args.config), "backbones", config["backbone"] + ".yaml")
        if os.path.exists(backbone_config_path):
            with open(backbone_config_path, "r") as f:
                backbone_config = yaml.safe_load(f)
                print("Loaded backbone config from", backbone_config_path)
            config = config | backbone_config
            config["backbone_config"]["input_channels"] = config["input_channels"]
        else:
            print("No additional config file found for backbone", config["backbone"])
    if "fusion" in config:
        # Load fusion config and add
        fusion_config_path = os.path.join(os.path.dirname(args.config), "fusion", config["fusion"] + ".yaml")
        if os.path.exists(fusion_config_path):
            with open(fusion_config_path, "r") as f:
                fusion_config = yaml.safe_load(f)
                print("Loaded fusion config from", fusion_config_path)
            config = config | fusion_config
        else:
            print("No additional config file found for fusion")

    # Calculate required shape for model inference
    max_factor_required = config["grid_size"]
    if "backbone_config" in config:
        max_factor_required = 2 ** (len(config["backbone_config"]["num_blocks"]) - 1) * \
                                config["backbone_config"]["stem"]["patch_size"] * \
                                np.max(config["backbone_config"]["stage"]["attention"]["partition_size"])
    elif "fusion_config" in config and "stem" in config["fusion_config"]:
        patch_size = config["fusion_config"]["stem"].get("patch_size", 2)
        partition_size = config["fusion_config"]["attention"].get("partition_size", 5)
        max_factor_required = 2 ** (len(config["fusion_config"]["num_blocks"]) - 1) * patch_size * partition_size

    # max_factor_required = 40 ###for all
    crop = np.array(ts_shape) % max_factor_required
    crop_mask = torch.ones(ts_shape, dtype=bool)
    crop_mask[:math.ceil(crop[0] / 2)] = False
    crop_mask[:, :math.ceil(crop[1] / 2)] = False
    if crop[0] > 1:
        crop_mask[-math.floor(crop[0] / 2):] = False
    if crop[1] > 1:
        crop_mask[:, -math.floor(crop[1] / 2):] = False
    cropped_shape = [ts_shape[0] - crop[0], ts_shape[1]- crop[1]]

    # Load model

    if config["train_type"] == "frame" or config["train_type"] == "event":
        config["input_channels"] = config["frame_input_channels"] if config["train_type"] == "frame" else config["event_input_channels"]
        model = SuperEvent(config).to(device)
    elif config["train_type"] == "fusion":
        config["augmentation"]["frame"]["enable_in_train"] = False
        config["augmentation"]["event"]["enable_in_train"] = False

        model = SuperEventFusion(config).to(device)
    elif config["train_type"] == "superpoint":
        raise ValueError("The legacy non-BN SuperPoint baseline is not included; use config/baselines/superpoint_bn.yaml.")
    elif config["train_type"] == "superpoint_bn":
        model = SuperPointBatchNormed(**{}).to(device)
    elif config["train_type"] == "aliked":
        from models.alike_nets.aliked import ALIKED
        model = ALIKED(top_k=np.max(cropped_shape) // 2, device=device)

    if config["train_type"] != "aliked":
        model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True),strict=False)
        model.to(device)
        model.eval()
        print("Loaded model weights from", args.model)

    if args.compile:
        model = torch.compile(model.to(torch.float16), backend="inductor", mode="max-autotune")
        print("Compiled model")
    
    if args.profile:
        from thop import profile
        # 模拟一个输入以统计 FLOPs
        # 根据你的 config 选择输入
        print("-" * 30)
        with torch.no_grad():
            # 构造 dummy input (形状需与 cropped_shape 一致)
            e_dummy = torch.randn(1, config["event_input_channels"], *cropped_shape).to(device)
            f_dummy = torch.randn(1, config["frame_input_channels"], *cropped_shape).to(device)

            if config["train_type"] == "fusion":
                model.frame_backbone.detector = nn.Identity()
                model.frame_backbone.descriptor = nn.Identity()
                input_data = (f_dummy, e_dummy)
            elif config["train_type"] in ["frame", "superpoint_bn"]:
                input_data = (f_dummy,)
            elif config["train_type"] == "aliked":
                input_data = (f_dummy.repeat(1,3,1,1),)
            else:
                input_data = (e_dummy,)

            flops, params = profile(model, inputs=input_data, verbose=False)

            print(f"Model Summary:")
            print(f"  - Parameters: {params / 1e6:.2f} M")
            print(f"  - FLOPs:      {flops / 1e9:.2f} G (at resolution {cropped_shape})")
        print("-" * 30)


if args.dataset_name == "mvsec":
    calib_path = os.path.join(args.calibration_root or os.path.join(args.dataset_path, "outdoor_night_calib"), "calib.txt")
# DVS 数据集稍后在循环内加载，这里设为 None
elif args.dataset_name == "dvs":
    calib_path = None
else:
    calib_path = os.path.join(args.calibration_root or args.dataset_path, "calib.txt")

if calib_path:
    calib = np.genfromtxt(calib_path)
    print(f"Loaded calibration from {calib_path}.")
    camera_matrix, distortion_coeffs = helpers.get_camera_matrix_and_distortion_coeffs(calib)

header = (
    f"{'Blur Thr':<10} | {'Med.Err':<8} | {'Inl.':<6} | {'Reproj':<6} | "
    f"{'AUC@5':<6} {'M.Err':<6} {'Cnt':<5} | "
    f"{'AUC@10':<6} {'M.Err':<6} {'Cnt':<5} | "
    f"{'AUC@20':<6} {'M.Err':<6} {'Cnt':<5} | "
    f"{'Total':<6}"
)

deg_intervals = np.array(range(0, 45, 1)) + 1
for sequence in args.sequence_names:
    print(f"\n{'='*40}")
    print(f"Processing Sequence: {sequence}")
    print(f"{'='*40}")

    if args.dataset_name == "dvs":
        # sequence 如 "bear/blur_1"，dirname 为 "bear"
        dataset_parent_dir = os.path.dirname(sequence)
        calib_path = os.path.join(args.dataset_path, dataset_parent_dir, "calib.txt")
        if os.path.exists(calib_path):
            calib = np.genfromtxt(calib_path)
            # 更新当前的相机内参
            camera_matrix, distortion_coeffs = helpers.get_camera_matrix_and_distortion_coeffs(calib)
        else:
            print(f"Warning: Calibration file not found at {calib_path}")

    # [新增] 为当前序列初始化结果字典
    sequence_summary = {thresh: {"r_error": [], "inlier_ratio": [], "num_matches":[], "num_inliers":[], "gt_rot": [], "reproj_error": [], } for thresh in blur_thresholds}
    # Load sequence data
    if args.dataset_name == "mvsec":
        hdf5_file_path = os.path.join(args.dataset_path, sequence, sequence + "_data.hdf5")
        gt_file_path = os.path.join(args.dataset_path, sequence, sequence + "_gt.hdf5")
        pair_file_path = os.path.join(args.dataset_path, sequence, sequence + "_final_indices.txt")
        calib_file_path = os.path.join(args.calibration_root or os.path.join(args.dataset_path, "outdoor_night_calib"), "camchain-imucam-outdoor_night.yaml")

        gt_file = h5py.File(gt_file_path, 'r', locking=False)
        poses = np.array(gt_file["davis"]["left"]["pose"])
        poses_timestamps = np.array(gt_file["davis"]["left"]["pose_ts"])

        h5_file = h5py.File(hdf5_file_path, 'r', locking=False)
        image = np.array(h5_file["davis"]["left"]["image_raw"])
        image_timestamps = np.array(h5_file["davis"]["left"]["image_raw_ts"])

        valid_images_indices = (image_timestamps >= poses_timestamps[0]) & (image_timestamps <= poses_timestamps[-1])
        image = image[valid_images_indices]
        image_timestamps = image_timestamps[valid_images_indices]

        calib_data = yaml.safe_load(open(calib_file_path, 'r'))
        sequence_K = np.eye(3)
        sequence_K[[0, 1, 0, 1], [0, 1, 2, 2]] = calib_data["cam0"]["intrinsics"]

        b_R = poses[:, :3, :3]
        b_t = poses[:, :3, 3]
        poses_interpolator = PoseInterpolator(timestamp=poses_timestamps, t=b_t, R=b_R, quat_R=False)

        image_data = {
            "image": image,
            "image_ts": image_timestamps,
            "K": sequence_K,
            "pose_interpolator": poses_interpolator,
            "length": len(image_timestamps)
        }
    else:
        sequence_path = os.path.join(args.dataset_path, sequence)
        poses = np.genfromtxt(os.path.join(sequence_path, pose_file_name))
        image_timestamps = np.genfromtxt(os.path.join(sequence_path, image_timestampes_file_name))
        if image_timestamps.ndim == 2:
            image_timestamps = image_timestamps[:, 0]
        frame_files = sorted(glob(os.path.join(sequence_path, frame_dir_name, "*.png")))
        pair_file_path = os.path.join(sequence_path, "final_indices.txt")
        image = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in frame_files])

        pose_timestamps = poses[:, 0]
        p_xyz = poses[:, 1:4]
        q_xyzw = poses[:, 4:]
        pose_interpolator = PoseInterpolator(timestamp=pose_timestamps, t=p_xyz, R=q_xyzw, quat_R=True, mode="linear")

        valid_images_indices = (image_timestamps >= pose_timestamps[0]) & (image_timestamps <= pose_timestamps[-1])
        image = image[valid_images_indices]
        image_timestamps = image_timestamps[valid_images_indices]

        image_data = {
            "image": image,
            "image_ts": image_timestamps,
            "K": camera_matrix,
            "pose_interpolator": pose_interpolator,
            "length": len(image_timestamps)
        }

        print(f"Loaded poses for sequence {sequence}.")

    if args.kpts_dir:
        if args.dataset_name == "eds":
            event_t = h5py.File(os.path.join(sequence_path, events_file_name), 'r', locking=False)["t"]
            first_event_time = event_t[0] * 1e-6
            last_event_time = event_t[-1] * 1e-6 - first_event_time
            poses[:, 0] -= first_event_time
        else: 
            # Avoid loading .txt, none of the approaches failed for this dataset
            last_event_time = np.inf
        
        baseline_name = os.path.basename(os.path.normpath(args.kpts_dir))
        if baseline_name == "RATE":
            distortion_coeffs = np.zeros(4)  # Haste (tracker in RATE) performs undistortion
            keypoints_path = os.path.join(args.kpts_dir, "tracks_" + sequence + ".txt")
            kpts = np.genfromtxt(keypoints_path)
        elif baseline_name == "LLAK":
            keypoints_path = os.path.join(args.kpts_dir, "tracks_" + sequence + ".csv")
            kpts = np.genfromtxt(keypoints_path, delimiter=',', skip_header=1)
            
            # Match format of RATE (id, t, x, y)
            kpts = np.vstack([kpts[:, 3], kpts[:, 2], kpts[:, 0], kpts[:, 1]]).T
            kpts[:, 1] *= 1e-6
        elif baseline_name in ["FEDETR", "FFKDT"]:
            keypoints_path = os.path.join(args.kpts_dir, sequence, sequence + ".csv")
            kpts = np.genfromtxt(keypoints_path, delimiter=',', skip_header=1)

            # Match format of RATE (id, t, x, y)
            kpts = np.vstack([kpts[:, 3], kpts[:, 2], kpts[:, 0], kpts[:, 1]]).T
            kpts[:, 1] *= 1e-6

        print(f"Loaded tracks of {np.max(kpts[:, 0])} keypoints from {keypoints_path}.")
        
        if args.dataset_name == "mvsec":
            start_time = max(poses_timestamps[0], kpts[np.argmax(kpts[:, 0] >= args.num_kpts_per_step), 1])
            end_time = min(poses_timestamps[-1], last_event_time)
            if args.num_kpts_per_step < 0:
                # Infer predicted keypoints per time step
                pose_frequency = (poses_timestamps[-1] - poses_timestamps[0]) / len(poses_timestamps)
                num_kpts_per_step = max(math.ceil(len(kpts) / (end_time - start_time) * pose_frequency), ts_shape[0] // 10)
                print(f"Assuming the latest {num_kpts_per_step} keypoint tracks to be active.")
        else:
            start_time = max(poses[0, 0], kpts[np.argmax(kpts[:, 0] >= args.num_kpts_per_step), 1])
            end_time = min(poses[-1, 0], last_event_time)
            if args.num_kpts_per_step < 0:
                # Infer predicted keypoints per time step
                pose_frequency = (poses[-1, 0] - poses[0, 0]) / len(poses)
                num_kpts_per_step = max(math.ceil(len(kpts) / (end_time - start_time) * pose_frequency), ts_shape[0] // 10)
                print(f"Assuming the latest {num_kpts_per_step} keypoint tracks to be active.")

        # Get keypoints with 
        if args.dataset_name == "ecd":
            # events = read_csv(os.path.join(sequence_path, events_file_name), header='infer', delimiter=" ", usecols=range(4)).to_numpy()
            events = np.load(os.path.join(sequence_path, events_file_name))
            start_time = poses[np.argmax(poses[:, 0] > events[0, 0].item() + np.max(args.model_delta_t)), 0]
            end_time = min(events[-1, 0].item(), poses[-1, 0])
        elif args.dataset_name == "dvs":
            events = np.load(os.path.join(sequence_path, events_file_name))
            events_x = events['x'].astype(np.float64)
            events_y = events['y'].astype(np.float64)
            events_t = events['t'].astype(np.float64)
            events_p = events['p'].astype(np.float64)
            events = np.stack([events_t, events_x, events_y, events_p], axis=1)
            start_time = poses[np.argmax(poses[:, 0] > events[0, 0].item() + np.max(args.model_delta_t)), 0]
            end_time = min(events[-1, 0].item(), poses[-1, 0])

        elif args.dataset_name == "eds":
            events = h5py.File(os.path.join(sequence_path, events_file_name), 'r', locking=False)
            os.path.join(os.path.join(sequence_path, events_file_name))
            start_time = poses[np.argmax(poses[:, 0] > events['t'][0] * 1e-6 + np.max(args.model_delta_t)), 0]
            end_time = min(events['t'][-1] * 1e-6, poses[-1, 0])
        elif args.dataset_name == "mvsec":
            events = h5_file["davis"]["left"]["events"]
            start_time = poses_timestamps[np.argmax(poses_timestamps[:] > events[0,2] + np.max(args.model_delta_t))]
            end_time = min(events[-1, 2], poses_timestamps[-1])
               
        print(f"Loaded events for sequence {sequence}.")

        pred_dict, ts_vis_dict, frame_vis_dict = process_tracked_keypoints(args, ts_shape, kpts, num_kpts_per_step, events, image_data, start_time, end_time, max_eval_delta_t=args.max_eval_delta_t)   


    else:
        # Get keypoints with 
        if args.dataset_name == "ecd":
            # events = read_csv(os.path.join(sequence_path, events_file_name), header='infer', delimiter=" ", usecols=range(4)).to_numpy()
            events = np.load(os.path.join(sequence_path, events_file_name))
            start_time = poses[np.argmax(poses[:, 0] > events[0, 0].item() + np.max(args.model_delta_t)), 0]
            end_time = min(events[-1, 0].item(), poses[-1, 0])
        elif args.dataset_name == "dvs":
            events = np.load(os.path.join(sequence_path, events_file_name))
            events_x = events['x'].astype(np.float64)
            events_y = events['y'].astype(np.float64)
            events_t = events['t'].astype(np.float64)
            events_p = events['p'].astype(np.float64)
            events = np.stack([events_t, events_x, events_y, events_p], axis=1)
            start_time = poses[np.argmax(poses[:, 0] > events[0, 0].item() + np.max(args.model_delta_t)), 0]
            end_time = min(events[-1, 0].item(), poses[-1, 0])

        elif args.dataset_name == "eds":
            events = h5py.File(os.path.join(sequence_path, events_file_name), 'r', locking=False)
            os.path.join(os.path.join(sequence_path, events_file_name))
            start_time = poses[np.argmax(poses[:, 0] > events['t'][0] * 1e-6 + np.max(args.model_delta_t)), 0]
            end_time = min(events['t'][-1] * 1e-6, poses[-1, 0])
        elif args.dataset_name == "mvsec":
            events = h5_file["davis"]["left"]["events"]
            start_time = poses_timestamps[np.argmax(poses_timestamps[:] > events[0,2] + np.max(args.model_delta_t))]
            end_time = min(events[-1, 2], poses_timestamps[-1])
               
        print(f"Loaded events for sequence {sequence}.")


        print(f"Running inference for sequence {sequence}...")
        pred_dict, ts_vis_dict, frame_vis_dict, model_inference_time, model_inference_iterations, nms_time, nms_iterations = \
            predict_keypoints(args, config, ts_shape, model, events, image_data,
                            start_time, end_time, cropped_shape=cropped_shape, compiled=args.compile)

        total_model_inference_time += model_inference_time
        total_model_inference_iterations += model_inference_iterations
        total_nms_time += nms_time
        total_nms_iterations += nms_iterations


    # 3. [修改] 多阈值循环评估
    for blur_thr in blur_thresholds:
        # print(f"\n--- Evaluating with Blur Threshold >= {blur_thr} deg ---") # [修改] 注释掉中间输出
        
        # 3.1 生成对应阈值的 Pair 文件名
        if args.dataset_name == "ecd":
            pair_file_name = f"{sequence}_blur_{blur_thr}.txt"
            pair_file_path = os.path.join(args.dataset_path, sequence, pair_file_name)
        elif args.dataset_name == "mvsec":
            pair_file_name = f"{sequence}_blur_{blur_thr}.txt"
            pair_file_path = os.path.join(args.dataset_path, sequence, pair_file_name)
        elif args.dataset_name == "dvs":
            pair_file_name = f"blur_{blur_thr}.txt"
            pair_file_path = os.path.join(args.dataset_path, sequence, pair_file_name)
        else:
            pair_file_name = f"{sequence}_blur_{blur_thr}.txt"
            pair_file_path = os.path.join(args.dataset_path, pair_file_name)
        
        if args.pairs_root:
            pair_file_path = os.path.join(args.pairs_root, sequence, pair_file_name)

        # Explicit reproduction runs must use the supplied fixed pairs.
        if args.require_existing_pairs and not os.path.isfile(pair_file_path):
            raise FileNotFoundError(f"Required fixed evaluation pairs not found: {pair_file_path}")

        # 3.2 如果文件不存在，则生成
        if not os.path.exists(pair_file_path):
            print(f"Generating pairs for threshold {blur_thr}...")
            indices = generate_pair_from_sequence_EC_double_blur(
                image_data, 
                time_window=7, 
                num_pairs=500, 
                min_inst_vel_deg=float(blur_thr), 
                max_baseline_rot_deg=15.0, 
                min_baseline_rot_deg=1.0
            ) 
            np.savetxt(pair_file_path, indices, fmt='%d')
        
        # 3.3 加载 Pair
        pair_indices_array = np.loadtxt(pair_file_path, dtype=int)
        if len(pair_indices_array.shape) == 1: 
             pair_indices_array = pair_indices_array.reshape(1, 2)
        # print(f"Loaded {len(pair_indices_array)} pairs for threshold {blur_thr}.") # [修改] 注释掉中间输出

        # 3.4 评估
        current_save_path = os.path.join(args.save_path, f"blur_{blur_thr}")
        os.makedirs(current_save_path, exist_ok=True)

        if sequence in ["poster_rotation", "shapes_rotation", "poster_6dof", "shapes_6dof", "hdr_poster"]:
            planar_sequence = True
        else:
            planar_sequence = False

        sequence_results = calculate_pose_estimation_error_pairs(
            pred_dict, 
            deg_intervals, 
            args.max_eval_delta_t, 
            pair_indices_array, 
            ts_vis_dict=ts_vis_dict,       
            frame_vis_dict=frame_vis_dict, 
            ransac_threshold=ransac_threshold, 
            img_shape=ts_shape, 
            device=device, 
            visualize=args.visualize,      
            save_path=current_save_path,
            dataset_name=args.dataset_name,
            is_planar=planar_sequence 
        )
        
        # 3.5 [修改] 将结果存入 **当前序列** 的摘要字典
        for key in ["r_error", "inlier_ratio", "num_matches","num_inliers","gt_rot", "reproj_error"]:
            sequence_summary[blur_thr][key].extend(sequence_results[key])
            # 如果你仍希望在最后打印所有序列的总表，取消下面这行的注释：
            final_summary[blur_thr][key].extend(sequence_results[key]) 

    # =========================================================
    # [新增] 这里是每个 Sequence 结束后的表格打印部分
    # =========================================================
    print(f"\nResults Table for Sequence: {sequence}")
    print("="*len(header))
    print(header)
    print("-" * len(header))

    for blur_thr in blur_thresholds:
        res = sequence_summary[blur_thr]
        
        # 如果该阈值下没有数据，跳过
        if not res["r_error"]:
            continue

        r_errors = np.array(res["r_error"])
        inliers = np.array(res["inlier_ratio"])
        reproj_errs = np.array(res["reproj_error"])
        
        # 1. 基础统计
        median_error = np.median(r_errors)
        avg_inlier = np.mean(inliers)

        success_mask = r_errors < 20.0
        if np.sum(success_mask) > 0:
            avg_reproj = np.mean(reproj_errs[success_mask])
        else:
            avg_reproj = 99.9

        num_samples = len(r_errors)
        
        # 2. 计算 AUC (静默模式)
        temp_results = {"r_error": res["r_error"], "inlier_ratio": res["inlier_ratio"], "gt_rot": res["gt_rot"]}
        original_stdout = sys.stdout
        sys.stdout = io.StringIO() # 屏蔽输出
        try:
            current_aucs = calculate_auc(temp_results, args.auc_thresholds)
        except:
            current_aucs = [0.0, 0.0, 0.0]
        finally:
            sys.stdout = original_stdout # 恢复输出
        
        final_AUC[blur_thr]["AUC@5"].append(current_aucs[0])
        final_AUC[blur_thr]["AUC@10"].append(current_aucs[1])
        final_AUC[blur_thr]["AUC@20"].append(current_aucs[2])

        # 3. 计算阈值内的 Mean Error 和 Count
        thresh_stats = []
        for thr in [5.0, 10.0, 20.0]:
            mask = r_errors <= thr
            count = np.sum(mask)
            mean_err = np.mean(r_errors[mask]) if count > 0 else 0.0
            thresh_stats.append((mean_err, count))
        
        # 4. 格式化打印行
        row_str = (
            f">= {blur_thr:<7} | "
            f"{median_error:<8.4f} | "
            f"{avg_inlier:<6.4f} | "
            f"{avg_reproj:<6.3f} | "
            f"{current_aucs[0]:<6.4f} {thresh_stats[0][0]:<6.4f} {thresh_stats[0][1]:<5d} | "
            f"{current_aucs[1]:<6.4f} {thresh_stats[1][0]:<6.4f} {thresh_stats[1][1]:<5d} | "
            f"{current_aucs[2]:<6.4f} {thresh_stats[2][0]:<6.4f} {thresh_stats[2][1]:<5d} | "
            f"{num_samples:<6d}"
        )
        print(row_str)

    print("="*len(header))
    print("\n")

# # ================== 最终结果汇总输出_new ==================
# # 定义表头 (为了防止换行，使用了较紧凑的缩写)
# # Thr: Threshold
# # Med.: Median Error
# # Inl.: Inlier Ratio
# # M.Err: Mean Error within threshold
# # Cnt: Count within threshold
header_final = (
    f"{'Blur Thr':<10}|"
    f"{'Inl_ra':<6}|"
    f"{'num_ma':<6}|"
    f"{'num_in':<6}|"
    f"{'AUC@5':<6}|"
    f"{'AUC@10':<6}|"
    f"{'AUC@20':<6}|"
)

print(header_final)
print("-" * len(header_final))

for blur_thr in blur_thresholds:

    current_aucs = [
        np.mean(final_AUC[blur_thr]["AUC@5"]),
        np.mean(final_AUC[blur_thr]["AUC@10"]),
        np.mean(final_AUC[blur_thr]["AUC@20"])
    ]

    current_inliers = [np.mean(final_summary[blur_thr]["inlier_ratio"]),
                       np.mean(final_summary[blur_thr]["num_matches"]),
                       np.mean(final_summary[blur_thr]["num_inliers"])
                       ]

    row_str = (
        f">= {blur_thr:<7} | " 
        f"{current_inliers[0]:<6.4f}| "
        f"{current_inliers[1]:<6.4f}| "
        f"{current_inliers[2]:<6.4f}| "
        
        # @5 deg
        f"{current_aucs[0]:<6.4f}| "
        
        # @10 deg
        f"{current_aucs[1]:<6.4f}| "
        
        # @20 deg
        f"{current_aucs[2]:<6.4f}| "
    )
    
    print(row_str)

print("="*len(header))

## 在你原来的代码里加上这段，只取 AUC@10 作为画图代表
print("--- 用于画图的数据 ---")
plot_data = []
for blur_thr in blur_thresholds:
    # 直接提取长度为 5 的列表，不求平均
    plot_data.append(final_AUC[blur_thr]["AUC@10"]) 
plot_data = np.round(np.array(plot_data).T * 100, 2).tolist()
print(f"Ours_AUC10 = {plot_data}")

# 在脚本最后
if total_model_inference_iterations > 0:
    final_avg_infer = (total_model_inference_time / total_model_inference_iterations) * 1000
    final_avg_nms = (total_nms_time / total_nms_iterations) * 1000
    
    print("\n" + " GLOBAL PROFILING ".center(40, "="))
    print(f"Total Frames Processed: {total_model_inference_iterations}")
    print(f"Avg Model Inference:    {final_avg_infer:.2f} ms")
    print(f"Avg NMS Time:           {final_avg_nms:.2f} ms")
    print(f"Avg Overall FPS:        {1000 / (final_avg_infer + final_avg_nms):.2f}")
    print("=" * 40)

# # ================== 最终结果汇总输出 ==================
# # 定义表头 (为了防止换行，使用了较紧凑的缩写)
# # Thr: Threshold
# # Med.: Median Error
# # Inl.: Inlier Ratio
# # M.Err: Mean Error within threshold
# # Cnt: Count within threshold

# print("\n" + "="*len(header))
# print(header)
# print("-" * len(header))

# for blur_thr in blur_thresholds:
#     res = final_summary[blur_thr]
    
#     r_errors = np.array(res["r_error"])
#     inliers = np.array(res["inlier_ratio"])
#     reproj_errs = np.array(res["reproj_error"]) # [新增]
    
#     # 1. 基础统计
#     median_error = np.median(r_errors)
#     avg_inlier = np.mean(inliers)

#     # 只统计成功帧的重投影误差 (error < 20 deg)，避免失败帧(100px误差)拉高平均值
#     success_mask = r_errors < 20.0
#     if np.sum(success_mask) > 0:
#         avg_reproj = np.mean(reproj_errs[success_mask])
#     else:
#         avg_reproj = 99.9

#     num_samples = len(r_errors)
    
#     # 2. 计算 AUC (这里假设 calculate_auc 返回 [auc5, auc10, auc20])
#     # 为了静默计算，我们不直接打印，而是构造 temp_results
#     temp_results = {"r_error": res["r_error"], "inlier_ratio": res["inlier_ratio"], "gt_rot": res["gt_rot"]}
    
#     # 临时重定向 stdout 防止 calculate_auc 内部 print 刷屏 (可选，如果 calculate_auc 很吵的话)
#     import sys, io
#     original_stdout = sys.stdout
#     sys.stdout = io.StringIO() 
#     try:
#         current_aucs = calculate_auc(temp_results, args.auc_thresholds)
#     except:
#         current_aucs = [0.0, 0.0, 0.0]
#     finally:
#         sys.stdout = original_stdout

#     # 3. 计算阈值内的 Mean Error 和 Count
#     # 阈值列表对应 AUC 的 [5, 10, 20]
#     thresh_stats = []
#     for thr in [5.0, 10.0, 20.0]:
#         mask = r_errors <= thr
#         count = np.sum(mask)
#         mean_err = np.mean(r_errors[mask]) if count > 0 else 0.0
#         thresh_stats.append((mean_err, count))
    
#     # 4. 格式化打印行
#     # 数据解包: (mean5, cnt5), (mean10, cnt10), (mean20, cnt20) = thresh_stats
    
#     row_str = (
#         f">= {blur_thr:<7} | "
#         f"{median_error:<8.4f} | "
#         f"{avg_inlier:<6.4f} | "
#         f"{avg_reproj:<6.3f} | " # [新增] 打印重投影误差
        
#         # @5 deg
#         f"{current_aucs[0]:<6.4f} {thresh_stats[0][0]:<6.4f} {thresh_stats[0][1]:<5d} | "
        
#         # @10 deg
#         f"{current_aucs[1]:<6.4f} {thresh_stats[1][0]:<6.4f} {thresh_stats[1][1]:<5d} | "
        
#         # @20 deg
#         f"{current_aucs[2]:<6.4f} {thresh_stats[2][0]:<6.4f} {thresh_stats[2][1]:<5d} | "
        
#         f"{num_samples:<6d}"
#     )
    
#     print(row_str)

# print("="*len(header))


