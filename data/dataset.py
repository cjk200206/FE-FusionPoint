import cv2
from enum import Enum
from glob import glob
import numpy as np
import os
import random

from torch.utils.data import Dataset
import torch

from data.homographies import sample_homography_corners, warp_points
from data.util import get_seq_and_idx, create_kp_maps, create_tencode_from_ts
from data_preparation.util.data_io import load_ts_sparse
from data.augment import ImageArrayAugmentation, modality_dropout_weighted

DataSplit = Enum("DataSplit", ["train", "val", "test"])

class DatasetCollection(Dataset):
    # Remark: This Class handles shuffeling for itself to prevent dimensionality problems
    # when batching (shuffle must be set to False in DataLoader when batch_size > 1)

    def __init__(self, data_split, config, skip_to_idx=0, frames=False, vis_mode=False, demo=False):
        self.skip_to_idx = skip_to_idx

        # Get dataset names
        assert data_split in DataSplit
        if data_split == DataSplit.train:
            self.dataset_names_list = config["dataset_names"]
        if data_split == DataSplit.val:
            self.dataset_names_list = [next(iter(name)) for name in config["val_sequences"]]
        if data_split == DataSplit.test:
            self.dataset_names_list = [next(iter(name)) for name in config["test_sequences"]]

        # Create sub-datasets
        self.dataset_configs_list = [{**config, "dataset_names": name} for name in self.dataset_names_list]
        self.dataset_list = [TsDataset(data_split, config, frames=frames, vis_mode=vis_mode, demo=demo) for config in self.dataset_configs_list]

        # Save useful data
        self.num_datasets = len(self.dataset_list)

        # Ignore elements left after batching
        self.dataset_lengths = [len(dataset) - len(dataset) % config["batch_size"] for dataset in self.dataset_list]
        self.num_elements = sum(self.dataset_lengths)
        self.num_batches_per_dataset = np.array(self.dataset_lengths) / config["batch_size"]

        # Generate random mapping of all dataset elements but with batches from same dataset
        batch_idx_arr = np.arange(sum(self.num_batches_per_dataset), dtype=int)
        
        if config["shuffle_data"]:
            np.random.shuffle(batch_idx_arr)

        # Convert random order batch indexes to sample indexes
        batch_idx_arr = batch_idx_arr * config["batch_size"]
        self.idx_map = np.concatenate([np.arange(i, i + config["batch_size"]) for i in batch_idx_arr]).astype(int)

        assert all(np.sort(self.idx_map) == np.arange(self.num_elements)), "Some elements are occurring multiple times or are missing."

        print(f"\nData statistics ({data_split.name}):")
        for i, dataset_name in enumerate(self.dataset_names_list):
            print(f"Dataset {dataset_name} contains {self.dataset_lengths[i]} samples ({100 * self.dataset_lengths[i] / self.num_elements} %).")

    def __len__(self):
        return self.num_elements
    
    def __getitem__(self, idx):
        # Prevent data I/O overhead if training is resumed
        if idx < self.skip_to_idx:
            return []
        else:
             self.skip_to_idx = 0

        idx = self.idx_map[idx]
        seq, idx = get_seq_and_idx(idx, self.dataset_lengths)
        return self.dataset_list[seq][idx]

class TsDataset(Dataset):
    def __init__(self, data_split, config, frames=False, vis_mode=False, demo=False):
        self.config = config
        self.vis_mode = vis_mode
        self.frames = frames

        if self.vis_mode:
            self.frames = False  # vis_mode has specific output

        assert data_split in DataSplit
        train_data_path = os.path.expanduser(self.config["train_data_path"])  # expand tilde
        data_base_path = os.path.join(train_data_path, data_split.name, self.config["dataset_names"])
        assert os.path.exists(data_base_path), data_base_path

        # Get paths of all sequences
        self.sequence_paths = [f.path for f in os.scandir(data_base_path) if f.is_dir()]
        if config.get("max_sequences_per_dataset", 0):
            self.sequence_paths = sorted(self.sequence_paths)[:config["max_sequences_per_dataset"]]

        # Create lists of all time surfaces
        self.ts_seq_list = [sorted(
            [p for p in glob(os.path.join(seq, "time_surfaces", "*.npz")) if "_" not in os.path.basename(p)]
            ) for seq in self.sequence_paths]
        
        if self.config["dataset_names"] in ["uevd", "gen_coco", "gen_coco_1"]:
            self.ts_clear_seq_list = [sorted(
                [p for p in glob(os.path.join(seq, "time_surfaces_clear", "*.npz")) if "_" not in os.path.basename(p)]
                ) for seq in self.sequence_paths]
            self.ts_heavy_blur_seq_list = [sorted(
                [p for p in glob(os.path.join(seq, "time_surfaces_heavy_blur", "*.npz")) if "_" not in os.path.basename(p)]
                ) for seq in self.sequence_paths]

        # Create lists of all match and valid indeces files
        if demo:
            self.kp_seq_list = [sorted(glob(os.path.join(seq, "demo_matches", "*.npz"))) for seq in self.sequence_paths]
        else:
            self.kp_seq_list = [sorted(glob(os.path.join(seq, "sg_matches", "*.npz"))) for seq in self.sequence_paths]
            if self.config["dataset_names"] == "gopro":
                self.kp_seq_list = [sorted(glob(os.path.join(seq, "low_res_sg_matches", "*.npz"))) for seq in self.sequence_paths]


        # Reduce to maximal allowed number of samples per sequence
        max_num_samples_per_sequence = -1
        if data_split == DataSplit.train and self.config["dataset_names"] in self.config["max_num_samples_per_sequence_train"].keys():
            max_num_samples_per_sequence = self.config["max_num_samples_per_sequence_train"][self.config["dataset_names"]]
        elif data_split == DataSplit.val and self.config["dataset_names"] in self.config["max_num_samples_per_sequence_val"].keys():
            max_num_samples_per_sequence = self.config["max_num_samples_per_sequence_val"][self.config["dataset_names"]]

        if max_num_samples_per_sequence > 0:
            self.kp_seq_list = [seq if len(seq) < max_num_samples_per_sequence 
                                else np.array(seq)[(np.arange(max_num_samples_per_sequence) * (len(seq) / max_num_samples_per_sequence)).astype(int).tolist()]
                                for seq in self.kp_seq_list]
        if not self.config["temporal_matching"]["enable"]:
            # Filter out duplicate time surfaces (only first ts used)
            # Filter out all occurances of the same id before the underscore
            self.kp_seq_list = [[path for i, path in enumerate(seq) 
                                 if i == 0 or os.path.basename(path).split('_')[0] != os.path.basename(seq[i - 1]).split('_')[0]]
                                for seq in self.kp_seq_list]

        # Compute sequence length
        self.num_samples_per_seq_list = [len(seq) for seq in self.kp_seq_list]
        self.num_elements = sum(self.num_samples_per_seq_list)

        # Generate random mapping of all dataset elements if desired
        self.index_mapping = np.arange(sum(self.num_samples_per_seq_list), dtype=int)
        if config["shuffle_data"]:
            np.random.shuffle(self.index_mapping)
        
        # Augment frames for training
        self.is_augment_frame = False
        self.is_augment_event = False
        if "augmentation" in self.config:
            self.is_augment_frame = self.config["augmentation"]["frame"]["enable"]
            self.augment_image = ImageArrayAugmentation(**self.config["augmentation"]["frame"]["params"])
            self.is_augment_event = self.config["augmentation"]["event"]["enable"]
            self.augment_event = ImageArrayAugmentation(**self.config["augmentation"]["event"]["params"], dropout_fill=0)

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        idx = self.index_mapping[idx]
        seq, seq_idx = get_seq_and_idx(idx, self.num_samples_per_seq_list)

        # Load matches (pseudo ground truth)
        matches = dict(np.load(self.kp_seq_list[seq][seq_idx], allow_pickle=True))

        # Time surface (input)
        ts0 = load_ts_sparse(self.ts_seq_list[seq][int(matches["image_id0"])]).astype(np.float32)
        if self.config["dataset_names"]in ["uevd", "gen_coco", "gen_coco_1"]:
            ts0_clear = load_ts_sparse(self.ts_clear_seq_list[seq][int(matches["image_id0"])]).astype(np.float32)
            ts0_heavy_blur = load_ts_sparse(self.ts_heavy_blur_seq_list[seq][int(matches["image_id0"])]).astype(np.float32)

        if self.config["temporal_matching"]["enable"]:
            ts1 = load_ts_sparse(self.ts_seq_list[seq][int(matches["image_id1"])]).astype(np.float32)
            if self.config["dataset_names"] in ["uevd", "gen_coco", "gen_coco_1"]:
                ts1_clear = load_ts_sparse(self.ts_clear_seq_list[seq][int(matches["image_id1"])]).astype(np.float32)
                ts1_heavy_blur = load_ts_sparse(self.ts_heavy_blur_seq_list[seq][int(matches["image_id1"])]).astype(np.float32)

        elif self.config.get("clear_blur_matching", {}).get("enable", False) and self.config["dataset_names"] in ["uevd", "gen_coco", "gen_coco_1"]:
            # Use ts0_clear and match with itself after homographic adaption
            ts1 = ts0.copy()
            ts1_clear = ts0_clear.copy()
            matches["keypoints1"] = matches["keypoints0"].copy()
            matches["matches0"] = np.arange(len(matches["keypoints0"]))
            matches["image_id1"] = matches["image_id0"]
        else:
            # Only use ts0 and and match with itself after homographic adaption
            ts1 = ts0.copy()
            matches["keypoints1"] = matches["keypoints0"].copy()
            matches["matches0"] = np.arange(len(matches["keypoints0"]))

        if self.config["dataset_names"] == "gopro":
            # Rescale time surfaces to original GoPro resolution
            ts0 = cv2.resize(ts0, (480, 320), interpolation=cv2.INTER_NEAREST)
            ts1 = cv2.resize(ts1, (480, 320), interpolation=cv2.INTER_NEAREST)

        # 3. Load Frames (Moved UP before augmentation)
        # We need to load frames now so we can warp them together with TS if needed
        frame0, frame1 = None, None
        f0, f1, f0_raw, f1_raw = None, None, None, None

        if self.vis_mode or self.frames:
            # Load grayscale frames
            # assert not self.config["homography_adaptation"]["enable"], "Not supported."
            frame_dir = os.path.join(os.path.dirname(os.path.dirname(self.ts_seq_list[seq][int(matches["image_id0"])])), "frames")
            
            if self.config.get("clear_blur_matching", {}).get("enable", False):
                frame0_path = os.path.join(frame_dir, str(matches["image_id0"]) + ".png")
                frame1_path = frame0_path
            else:
                frame0_path = os.path.join(frame_dir, str(matches["image_id0"]) + ".png")
                frame1_path = os.path.join(frame_dir, str(matches["image_id1"]) + ".png")

            if self.config["dataset_names"] == "sde":
                low_frame_dir = os.path.join(os.path.dirname(os.path.dirname(self.ts_seq_list[seq][int(matches["image_id0"])])), "low_frames")
                low_frame_path0 = os.path.join(low_frame_dir, str(matches["image_id0"]) + ".png")   
                low_frame_path1 = os.path.join(low_frame_dir, str(matches["image_id1"]) + ".png")
                low_frame0 = cv2.imread(low_frame_path0, )
                low_frame1 = cv2.imread(low_frame_path1)
                frame0 = cv2.imread(frame0_path)
                frame1 = cv2.imread(frame1_path)

                if self.config["augmentation"].get("low_image", {}).get("enable", False):
                    if random.random() < self.config["augmentation"]["low_image"]["p"]:
                        frame0 = low_frame0
                    # if random.random() < self.config["augmentation"]["low_image"]["p"]:
                        frame1 = low_frame1

            elif self.config["dataset_names"] in ["gopro", "uevd","gen_coco", "gen_coco_1"]:
                sharp_frame_dir = os.path.join(os.path.dirname(os.path.dirname(self.ts_seq_list[seq][int(matches["image_id0"])])), "sharp_frames")
                
                if self.config.get("clear_blur_matching", {}).get("enable", False):
                    sharp_frame_path0 = os.path.join(sharp_frame_dir, str(matches["image_id0"]) + ".png")
                    sharp_frame_path1 = sharp_frame_path0
                else:
                    sharp_frame_path0 = os.path.join(sharp_frame_dir, str(matches["image_id0"]) + ".png")
                    sharp_frame_path1 = os.path.join(sharp_frame_dir, str(matches["image_id1"]) + ".png")

                if self.config["dataset_names"] in ["uevd", "gen_coco", "gen_coco_1"]:
                    if self.config["augmentation"].get("use_real_blur", False):
                        blur_5_frame_path0 = sharp_frame_path0.replace("sharp_frames", "blur_5_frames")
                        blur_7_frame_path0 = sharp_frame_path0.replace("sharp_frames", "blur_7_frames")
                        blur_9_frame_path0 = sharp_frame_path0.replace("sharp_frames", "blur_9_frames")
                        blur_5_frame_path1 = sharp_frame_path1.replace("sharp_frames", "blur_5_frames")
                        blur_7_frame_path1 = sharp_frame_path1.replace("sharp_frames", "blur_7_frames")
                        blur_9_frame_path1 = sharp_frame_path1.replace("sharp_frames", "blur_9_frames")

                sharp_frame0 = cv2.imread(sharp_frame_path0)
                sharp_frame1 = cv2.imread(sharp_frame_path1)
                frame0 = sharp_frame0
                frame1 = sharp_frame1

                ts_flag = False
                ts0_flag = False
                ts1_flag = False
                blur_flag = False
                if self.config["augmentation"].get("blur_image", {}).get("enable", False):  
                    if self.config.get("clear_blur_matching", {}).get("enable", False):
                        if random.random() < self.config["augmentation"]["blur_image"]["p"]:
                            frame0 = cv2.imread(frame0_path)
                            ts0_flag = True
                        # if random.random() < self.config["augmentation"]["blur_image"]["p"]:
                            frame1 = cv2.imread(frame1_path)
                            ts1_flag = True
                        
                        if not ts0_flag:
                            ts0 = ts0_clear if self.config["dataset_names"] in ["uevd", "gen_coco", "gen_coco_1"] else np.zeros_like(ts0)
                        # if not ts1_flag:
                            ts1 = ts1_clear if self.config["dataset_names"] in ["uevd", "gen_coco", "gen_coco_1"] else np.zeros_like(ts1)
                            
                    else:
                        if random.random() < self.config["augmentation"]["blur_image"]["p"]:
                            if self.config["dataset_names"] in ["uevd", "gen_coco", "gen_coco_1"] and self.config["augmentation"].get("use_real_blur", False):
                                blur_type_weights = [4,2,2,2]
                                blur_type = random.choices(["blur","blur_5", "blur_7", "blur_9"], weights=blur_type_weights)[0]
                                blur_flag = True

                                if blur_type == "blur_5":
                                    frame0 = cv2.imread(blur_5_frame_path0)
                                    frame1 = cv2.imread(blur_5_frame_path1)
                                elif blur_type == "blur_7":
                                    frame0 = cv2.imread(blur_7_frame_path0)
                                    frame1 = cv2.imread(blur_7_frame_path1)
                                    ts0 = ts0_heavy_blur
                                    ts1 = ts1_heavy_blur
                                elif blur_type == "blur_9":
                                    frame0 = cv2.imread(blur_9_frame_path0)
                                    frame1 = cv2.imread(blur_9_frame_path1)
                                    ts0 = ts0_heavy_blur
                                    ts1 = ts1_heavy_blur
                                else:
                                    frame0 = cv2.imread(frame0_path)
                                    frame1 = cv2.imread(frame1_path)
                            else:
                                frame0 = cv2.imread(frame0_path)
                                frame1 = cv2.imread(frame1_path)
                                blur_flag = True

                            ts_flag = True
                        if not ts_flag:
                            ts0 = np.zeros_like(ts0)
                            ts1 = np.zeros_like(ts1)
                            if self.config["dataset_names"] in ["uevd", "gen_coco", "gen_coco_1"]:
                                ts0 = ts0_clear
                                ts1 = ts1_clear
                        
                if self.config["dataset_names"] == "gopro":
                    frame0 = cv2.resize(frame0, (480, 320), interpolation=cv2.INTER_AREA)
                    frame1 = cv2.resize(frame1, (480, 320), interpolation=cv2.INTER_AREA)
                    sharp_frame0 = cv2.resize(sharp_frame0, (480, 320), interpolation=cv2.INTER_AREA)
                    sharp_frame1 = cv2.resize(sharp_frame1, (480, 320), interpolation=cv2.INTER_AREA)

            else:
                frame0 = cv2.imread(frame0_path)
                frame1 = cv2.imread(frame1_path)


        # [新增] 初始化 pixel_mapping
        pixel_mapping = None 
        
        if self.config["homography_adaptation"]["enable"]:
            ts_shape = ts0.shape[:2][::-1] # (W, H)
            
            # [新增] 生成原始坐标网格 (Coordinate Grid)
            if self.config.get("clear_blur_matching", {}).get("enable", False):
                # meshgrid 生成的是 (H, W) 的矩阵
                grid_x, grid_y = np.meshgrid(np.arange(ts_shape[0]), np.arange(ts_shape[1]))
                # 堆叠成 (H, W, 2), 值是 float 类型以便计算
                grid = np.stack((grid_x, grid_y), axis=-1).astype(np.float32)
                
            H, _, _, _ = sample_homography_corners(ts_shape,
                                                ts_shape,
                                                difficulty=self.config["homography_adaptation"]["difficulty"],
                                                max_angle=self.config["homography_adaptation"]["max_angle"])
            
            ts0 = cv2.warpPerspective(ts0, H, ts_shape)
            matches["keypoints0"] = warp_points(matches["keypoints0"], H, inverse=False)
            
            # [新增] 对坐标网格进行同样的 Warp
            # 结果含义：pixel_mapping[y, x] = [src_x, src_y]
            # 即变换后图像在 (x,y) 处的像素，对应变换前图像的 (src_x, src_y)
            if self.config.get("clear_blur_matching", {}).get("enable", False):
                pixel_mapping = cv2.warpPerspective(grid, H, ts_shape, flags=cv2.INTER_LINEAR)

            # Warp Frame0 if it exists
            if frame0 is not None:
                frame0 = cv2.warpPerspective(frame0, H, ts_shape)
                if self.config["dataset_names"] in ["gopro", "uevd", "gen_coco", "gen_coco_1"]:
                    sharp_frame0 = cv2.warpPerspective(sharp_frame0, H, ts_shape)

            # Rescale time surface if required
            if self.config["homography_adaptation"]["resize_factor"] > 0:
                rf = self.config["homography_adaptation"]["resize_factor"]
                ts0 = cv2.resize(ts0, (0,0), fx=rf, fy=rf, interpolation=cv2.INTER_NEAREST)
                ts1 = cv2.resize(ts1, (0,0), fx=rf, fy=rf, interpolation=cv2.INTER_NEAREST)
                matches["keypoints0"] = matches["keypoints0"] * rf
                matches["keypoints1"] = matches["keypoints1"] * rf
                
                # [新增] 对 Mapping 进行 Resize
                if pixel_mapping is not None:
                    # Resize 网格图本身
                    pixel_mapping = cv2.resize(pixel_mapping, (0,0), fx=rf, fy=rf, interpolation=cv2.INTER_LINEAR)
                    # 关键：网格里的“值”是原始坐标，如果原始图像也被理解为缩放了，
                    # 那么值也需要乘以 rf 才能对应到缩放后的"原始"图像坐标系。
                    # 通常我们希望对应到"本次变换的输入"（即未缩放的原始图），则不需要乘。
                    # 但考虑到 keypoints0 也乘以了 rf，为了保持一致性（坐标系统一），这里也应该乘 rf。
                    pixel_mapping = pixel_mapping * rf

                if frame0 is not None:
                    frame0 = cv2.resize(frame0, (0,0), fx=rf, fy=rf, interpolation=cv2.INTER_AREA)
                    frame1 = cv2.resize(frame1, (0,0), fx=rf, fy=rf, interpolation=cv2.INTER_AREA)
                    if self.config["dataset_names"] in ["gopro", "uevd", "gen_coco", "gen_coco_1"]:
                        sharp_frame0 = cv2.resize(sharp_frame0, (0,0), fx=rf, fy=rf, interpolation=cv2.INTER_AREA)
                        sharp_frame1 = cv2.resize(sharp_frame1, (0,0), fx=rf, fy=rf, interpolation=cv2.INTER_AREA)
                    
            matches["keypoints0"] = np.rint(matches["keypoints0"]).astype(int)
            matches["keypoints1"] = np.rint(matches["keypoints1"]).astype(int)

        # 5. Identifier generation (Moved here)
        identifier = ""
        if self.vis_mode or self.frames:
            dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.ts_seq_list[seq][int(matches["image_id0"])]))))
            sequence_name = os.path.basename(os.path.dirname(os.path.dirname(self.ts_seq_list[seq][int(matches["image_id0"])])))
            identifier = dataset_name + "_" + sequence_name + "_" + str(matches["image_id0"]) + "_" + str(matches["image_id1"])
        
        # Filter all keypoints to lie within the image bounds
        ts_shape = ts0.shape[:2][::-1]
        valid_mask_keypoints0 = np.all([matches["keypoints0"][:, 0] >= 0, matches["keypoints0"][:, 0] < ts_shape[0],
                                matches["keypoints0"][:, 1] >= 0, matches["keypoints0"][:, 1] < ts_shape[1]], axis=0)
        matches["keypoints0"] = matches["keypoints0"][valid_mask_keypoints0]
        matches["matches0"] = matches["matches0"][valid_mask_keypoints0]

        matches["keypoints1"][matches["keypoints1"][:, 0] > ts_shape[0]][:, 0] = ts_shape[0] - 1 
        matches["keypoints1"][matches["keypoints1"][:, 1] > ts_shape[1]][:, 1] = ts_shape[1] - 1

        # Sanity check
        if ts0.shape[:2][::-1] != ts1.shape[:2][::-1]:
            print(f"\n[WARNING] Skipping index {idx} : Shape mismatch ts0 vs ts1. "
                f"Got {ts0.shape} and {ts1.shape}.\n")
            return None
        elif self.vis_mode or self.frames:
            if ts0.shape[:2][::-1] != frame0.shape[:2][::-1]:
                print(f"\n[WARNING] Skipping index {idx} ({frame0_path}): Shape mismatch ts0 vs frame0. "
                    f"Got {ts0.shape[:2]} and {frame0.shape[:2]}.\n")
                return None
            if ts1.shape[:2][::-1] != frame1.shape[:2][::-1]:
                print(f"\n[WARNING] Skipping index {idx} ({frame1_path}): Shape mismatch ts1 vs frame1. "
                    f"Got {ts1.shape[:2]} and {frame1.shape[:2]}.\n")
                return None

        # Create keypoint maps
        kp_map0, kp_map1 = create_kp_maps(matches, ts0.shape[:2])

        if self.config["dataset_names"] == "gopro":
            # 目标尺寸 (H, W) = (240, 320)
            tgt_h, tgt_w = 240, 320
            curr_h, curr_w = ts0.shape[:2]  # numpy shape is (H, W)

            # 计算中心裁剪的起始坐标 (Center Crop)
            y_start = (curr_h - tgt_h) // 2
            x_start = (curr_w - tgt_w) // 2
            y_end = y_start + tgt_h
            x_end = x_start + tgt_w

            # 1. 裁剪 Time Surface
            ts0 = ts0[y_start:y_end, x_start:x_end]
            ts1 = ts1[y_start:y_end, x_start:x_end]

            # 2. 裁剪 Keypoint Map
            kp_map0 = kp_map0[y_start:y_end, x_start:x_end]
            kp_map1 = kp_map1[y_start:y_end, x_start:x_end]

            # 3. 裁剪 Frames (如果已加载)
            if self.vis_mode or self.frames:
                if frame0 is not None:
                    frame0 = frame0[y_start:y_end, x_start:x_end]
                    frame1 = frame1[y_start:y_end, x_start:x_end]
                
                # 处理 GoPro 特有的 sharp frames
                # 检查 sharp_frame 变量是否存在且非空
                if 'sharp_frame0' in locals() and sharp_frame0 is not None:
                    sharp_frame0 = sharp_frame0[y_start:y_end, x_start:x_end]
                    sharp_frame1 = sharp_frame1[y_start:y_end, x_start:x_end]
            
            if pixel_mapping is not None:
                pixel_mapping = pixel_mapping[y_start:y_end, x_start:x_end]
                # 注意：这里我们裁剪了视图，但存储的坐标值还是相对于左上角原点的。
                # 如果 keypoints0 减去了 (x_start, y_start)，为了让 mapping 的值
                # 也能和 keypoints0 在同一个坐标系下，Mapping 里的值也应该减去偏移量。
                pixel_mapping[..., 0] -= x_start
                pixel_mapping[..., 1] -= y_start

            # 4. 修正并过滤 Keypoints (非常重要！)
            # 坐标平移
            matches["keypoints0"][:, 0] -= x_start
            matches["keypoints0"][:, 1] -= y_start
            matches["keypoints1"][:, 0] -= x_start
            matches["keypoints1"][:, 1] -= y_start

            # 过滤超出新边界的点 (kp0)
            valid_mask0 = (matches["keypoints0"][:, 0] >= 0) & (matches["keypoints0"][:, 0] < tgt_w) & \
                          (matches["keypoints0"][:, 1] >= 0) & (matches["keypoints0"][:, 1] < tgt_h)
            matches["keypoints0"] = matches["keypoints0"][valid_mask0]
            if "matches0" in matches:
                matches["matches0"] = matches["matches0"][valid_mask0]

            # 过滤超出新边界的点 (kp1)
            valid_mask1 = (matches["keypoints1"][:, 0] >= 0) & (matches["keypoints1"][:, 0] < tgt_w) & \
                          (matches["keypoints1"][:, 1] >= 0) & (matches["keypoints1"][:, 1] < tgt_h)
            matches["keypoints1"] = matches["keypoints1"][valid_mask1]
            if "matches1" in matches:
                matches["matches1"] = matches["matches1"][valid_mask1]

        # Randomly crop to be devisible grid size
        max_factor_required = self.config["grid_size"]
        if "backbone_config" in self.config:
            max_factor_required = 2 ** (len(self.config["backbone_config"]["num_blocks"]) - 1) * \
                                self.config["backbone_config"]["stem"]["patch_size"] * \
                                np.max(self.config["backbone_config"]["stage"]["attention"]["partition_size"])
        elif "fusion_config" in self.config and "stem" in self.config["fusion_config"]:
            patch_size = self.config["fusion_config"]["stem"].get("patch_size", 2)
            partition_size = self.config["fusion_config"]["attention"].get("partition_size", 5)
            max_factor_required = 2 ** (len(self.config["fusion_config"]["num_blocks"]) - 1) * patch_size * partition_size
        required_crop = np.array(ts0.shape)[:-1] % max_factor_required
        random_crop = np.zeros(2, dtype=int)

        random_crop[required_crop > 0] = np.random.randint(0, high=required_crop[required_crop > 0])
        if required_crop[0] > random_crop[0]:
            ts0 = ts0[random_crop[0]:random_crop[0] - required_crop[0]]
            kp_map0 = kp_map0[random_crop[0]:random_crop[0] - required_crop[0]]
            # [新增] 随机裁剪 Mapping (H方向)
            if pixel_mapping is not None:
                pixel_mapping = pixel_mapping[random_crop[0]:random_crop[0] - required_crop[0]]
        else:
            ts0 = ts0[random_crop[0]:]
            kp_map0 = kp_map0[random_crop[0]:]
            # [新增] 随机裁剪 Mapping (H方向)
            if pixel_mapping is not None:
                pixel_mapping = pixel_mapping[random_crop[0]:]
                
        if required_crop[1] > random_crop[1]:
            ts0 = ts0[:, random_crop[1]:random_crop[1] - required_crop[1]]
            kp_map0 = kp_map0[:, random_crop[1]:random_crop[1] - required_crop[1]]
            # [新增] 随机裁剪 Mapping (W方向)
            if pixel_mapping is not None:
                pixel_mapping = pixel_mapping[:, random_crop[1]:random_crop[1] - required_crop[1]]
        else:
            ts0 = ts0[:, random_crop[1]:]
            kp_map0 = kp_map0[:, random_crop[1]:]
            # [新增] 随机裁剪 Mapping (W方向)
            if pixel_mapping is not None:
                pixel_mapping = pixel_mapping[:, random_crop[1]:]

        if self.vis_mode or self.frames:
            # Also crop frame0
            if required_crop[0] > random_crop[0]:
                frame0 = frame0[random_crop[0]:random_crop[0] - required_crop[0]]
            else:
                frame0 = frame0[random_crop[0]:]
            if required_crop[1] > random_crop[1]:
                frame0 = frame0[:, random_crop[1]:random_crop[1] - required_crop[1]]
            else:
                frame0 = frame0[:, random_crop[1]:]
            
            if self.config["dataset_names"] in ["gopro", "uevd", "gen_coco", "gen_coco_1"]:
                if required_crop[0] > random_crop[0]:
                    sharp_frame0 = sharp_frame0[random_crop[0]:random_crop[0] - required_crop[0]]
                else:
                    sharp_frame0 = sharp_frame0[random_crop[0]:]
                if required_crop[1] > random_crop[1]:
                    sharp_frame0 = sharp_frame0[:, random_crop[1]:random_crop[1] - required_crop[1]]
                else:
                    sharp_frame0 = sharp_frame0[:, random_crop[1]:]

        random_crop = np.zeros(2, dtype=int)
        random_crop[required_crop > 0] = np.random.randint(0, high=required_crop[required_crop > 0])
        if required_crop[0] > random_crop[0]:
            ts1 = ts1[random_crop[0]:random_crop[0] - required_crop[0]]
            kp_map1 = kp_map1[random_crop[0]:random_crop[0] - required_crop[0]]
        else:
            ts1 = ts1[random_crop[0]:]
            kp_map1 = kp_map1[random_crop[0]:]
        if required_crop[1] > random_crop[1]:
            ts1 = ts1[:, random_crop[1]:random_crop[1] - required_crop[1]]
            kp_map1 = kp_map1[:, random_crop[1]:random_crop[1] - required_crop[1]]
        else:
            ts1 = ts1[:, random_crop[1]:]
            kp_map1 = kp_map1[:, random_crop[1]:]

        if self.vis_mode or self.frames:
            # Also crop frame1
            if required_crop[0] > random_crop[0]:
                frame1 = frame1[random_crop[0]:random_crop[0] - required_crop[0]]
            else:
                frame1 = frame1[random_crop[0]:]
            if required_crop[1] > random_crop[1]:
                frame1 = frame1[:, random_crop[1]:random_crop[1] - required_crop[1]]
            else:
                frame1 = frame1[:, random_crop[1]:]
            
            if self.config["dataset_names"] in ["gopro", "uevd", "gen_coco", "gen_coco_1"]:
                if required_crop[0] > random_crop[0]:
                    sharp_frame1 = sharp_frame1[random_crop[0]:random_crop[0] - required_crop[0]]
                else:
                    sharp_frame1 = sharp_frame1[random_crop[0]:]
                if required_crop[1] > random_crop[1]:
                    sharp_frame1 = sharp_frame1[:, random_crop[1]:random_crop[1] - required_crop[1]]
                else:
                    sharp_frame1 = sharp_frame1[:, random_crop[1]:]

        # Convert to tencode if required
        if self.config["input_representation"] == "tencode":
            tencode_l0 = create_tencode_from_ts(ts0[..., 4], ts0[..., 9], 0.01)  # value only for ablation, 0.02 in paper
            tencode_m0 = create_tencode_from_ts(ts0[..., 4], ts0[..., 9], random.uniform(0.02, 0.035))
            tencode_h0 = create_tencode_from_ts(ts0[..., 4], ts0[..., 9], random.uniform(0.035, 0.05))
            tencode_l1 = create_tencode_from_ts(ts1[..., 4], ts1[..., 9], 0.01)  # value only for ablation, 0.02 in paper

            # Weird order is for capability with returned time surfaces when used for inference 
            return tencode_l0, tencode_l1, kp_map0, kp_map1, tencode_m0, tencode_h0

        # Change from channels last to channels first
        ts0 = ts0.transpose(2, 0, 1)
        ts1 = ts1.transpose(2, 0, 1)

        # [新增] 处理 Mapping 的维度 (H, W, 2) -> (2, H, W)
        if pixel_mapping is not None:
            pixel_mapping = pixel_mapping.transpose(2, 0, 1)

        if self.config["input_representation"] == "ts":
            ts0 = np.max([ts0[2], ts0[7]], axis=0)[None]
            ts1 = np.max([ts1[2], ts1[7]], axis=0)[None]
        elif self.config["input_representation"] == "mcts_1":
            ts0 = ts0[[2, 7]]
            ts1 = ts1[[2, 7]]
        elif self.config["input_representation"] == "mcts_3":
            ts0 = ts0[[1,2,3,6,7,8]] #[0.003,0.01,0.03]
            ts1 = ts1[[1,2,3,6,7,8]]
        
        if self.vis_mode or self.frames:
            f0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            f1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            if self.config["dataset_names"] in ["gopro", "uevd", "gen_coco", "gen_coco_1"]:
                f0_raw = cv2.cvtColor(sharp_frame0, cv2.COLOR_BGR2GRAY)
                f1_raw = cv2.cvtColor(sharp_frame1, cv2.COLOR_BGR2GRAY)
            else:   
                f0_raw = f0.copy()
                f1_raw = f1.copy()

            f0_mask = np.ones_like(f0, dtype=np.float32)
            f1_mask = np.ones_like(f1, dtype=np.float32)
            ts0_mask = np.ones_like(ts0, dtype=np.float32)[0:1]
            ts1_mask = np.ones_like(ts1, dtype=np.float32)[0:1]

            if self.is_augment_frame:
                # Augment frames
                # if self.config["dataset_names"]  not in ["gopro", "uevd", "gen_coco", "gen_coco_1"]:
                #     f0, f0_mask = self.augment_image(f0)
                #     f1, f1_mask = self.augment_image(f1)
                # elif not blur_flag:
                #     f0, f0_mask = self.augment_image(f0)
                #     f1, f1_mask = self.augment_image(f1)
                f0, f0_mask = self.augment_image(f0)
                f1, f1_mask = self.augment_image(f1)
            

            if self.is_augment_event:
                # Augment time surfaces
                ts0, ts0_mask = self.augment_event(ts0.transpose(1,2,0))
                ts1, ts1_mask = self.augment_event(ts1.transpose(1,2,0))
                ts0 = ts0.transpose(2,0,1)
                ts1 = ts1.transpose(2,0,1)
                ts0_mask = ts0_mask.transpose(2,0,1)
                ts1_mask = ts1_mask.transpose(2,0,1)
            
            if "augmentation" in self.config and self.config["augmentation"]["modality_drop"]["enable"]:
                # Apply modality dropout
                frame_drop_p = self.config["augmentation"]["modality_drop"]["frame_drop_p"]
                event_drop_p = self.config["augmentation"]["modality_drop"]["event_drop_p"]
                prob_list = [1-frame_drop_p-event_drop_p, event_drop_p, frame_drop_p]
                f0, ts0 = modality_dropout_weighted(f0, ts0, prob_list)
                f1, ts1 = modality_dropout_weighted(f1, ts1, prob_list)
        
        ts0_exsit_mask = np.sum(ts0, axis=0) > 0
        ts1_exsit_mask = np.sum(ts1, axis=0) > 0

        if self.vis_mode:
            output = {
                "ts0": ts0,
                "ts1": ts1,
                "kp_map0": kp_map0,
                "kp_map1": kp_map1,
                "frame0": frame0,
                "frame1": frame1,
                "identifier": identifier,
                "f0": f0,
                "f1": f1,
                "f0_raw": f0_raw,
                "f1_raw": f1_raw,
                "keypoints0": matches["keypoints0"],
                "keypoints1": matches["keypoints1"],
                "matches0": matches["matches0"],
                "matches1": matches["matches1"],
                }
        elif self.frames:
            output = {
                "ts0": ts0,
                "ts1": ts1,
                "kp_map0": kp_map0,
                "kp_map1": kp_map1,
                "f0": f0,
                "f1": f1,
                "f0_raw": f0_raw,
                "f1_raw": f1_raw,
                "ts0_mask": ts0_mask,
                "ts1_mask": ts1_mask,
                "f0_mask": f0_mask,
                "f1_mask": f1_mask,
                "ts0_exsit_mask": ts0_exsit_mask,
                "ts1_exsit_mask": ts1_exsit_mask,
                "keypoints0": matches["keypoints0"],
                "keypoints1": matches["keypoints1"],
                "matches0": matches["matches0"],
                "matches1": matches["matches1"],
            }
            # [新增] 将 mapping 加入输出
            if pixel_mapping is not None:
                output["pixel_mapping"] = pixel_mapping
        else:
            output = {
                "ts0": ts0,
                "ts1": ts1,
                "kp_map0": kp_map0,
                "kp_map1": kp_map1
            }
            # [新增] 将 mapping 加入输出
            if pixel_mapping is not None:
                output["pixel_mapping"] = pixel_mapping
        return output
    
def safe_collate_fn(batch):
    """
    一个自定义的 collate_fn，它会过滤掉值为 None 的样本。
    :param batch: Dataloader` 从 `Dataset` 的 `__getitem__` 中获取的样本列表。
    :return: 一个经过整理和堆叠的批次，或者在整个批次都无效时返回 None。
    """

    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None
    
    # 如果批次不为空，使用默认的 collate 函数来处理剩下的好数据
    return torch.utils.data.dataloader.default_collate(batch)