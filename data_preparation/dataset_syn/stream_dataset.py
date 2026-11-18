import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import cv2
import numpy as np

class SyntheticMotionStream(object):
    """
    更新版：支持多种运动模式（mixed, rotation），并将轨迹生成与渲染解耦。
    """
    def __init__(self, height, width, min_frames, max_frames, fps=200, mode='mixed'):
        self.height = height
        self.width = width
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.fps = fps
        self.mode = mode  # 'mixed' or 'rotation'
        self.dt = 1e6 

    def sample_trajectory_params(self, H_raw, W_raw):
        """
        根据 self.mode 生成起始和结束的四点坐标。
        如果需要手动控制参数，可以直接继承此类重写此方法，或直接构造点传给 render_sequence。
        Returns:
            pts_start: (4, 2)
            pts_end: (4, 2)
        """
        if self.mode == 'rotation':
            # === [核心修改] 将旋转拆分为 Roll, Yaw, Pitch ===
            # 定义基础框 (保持你原来的中心对齐逻辑)
            scale = 0.6
            w, h = int(W_raw * scale), int(H_raw * scale)
            x_c, y_c = W_raw / 2, H_raw / 2
            x_s, y_s = x_c - w / 2, y_c - h / 2
            
            pts_start = np.float32([
                [x_s, y_s], [x_s + w, y_s], 
                [x_s + w, y_s + h], [x_s, y_s + h]
            ])
            
            # 随机选择一种旋转主导模式，或者混合
            # 0: Roll (面内旋转) - 你原本的逻辑
            # 1: Yaw (左右扫) - 产生横向长拖影
            # 2: Pitch (上下扫) - 产生纵向长拖影
            rot_type = np.random.choice(['roll', 'yaw', 'pitch'], p=[1, 0, 0])

            pts_end = pts_start.copy()

            if rot_type == 'roll':
                # --- 你原本的代码 (面内旋转) ---
                rot_angle = np.random.uniform(-90, 90) 
                rot_mat = cv2.getRotationMatrix2D((x_c, y_c), rot_angle, 1.0)
                pts_end = cv2.transform(np.array([pts_start]), rot_mat)[0]

            elif rot_type == 'yaw':
                # --- [新增] 模拟摇头 (产生梯形畸变) ---
                # 左边边长变短/长，右边相反，模拟平面绕Y轴旋转
                
                # [修改] 限制最大畸变幅度，防止四边形交叉
                # 最大只允许缩放到原边长的 80% (即 squeeze < h/2 * 0.4)
                max_squeeze = h * 0.2 
                magnitude = np.random.uniform(h * 0.05, max_squeeze) 
                
                direction = np.random.choice([-1, 1])
                
                # 这是一个简单的透视模拟：让矩形变成梯形
                # 左右移动整个视窗来模拟大视角的扫视（配合畸变）
                shift = w * np.random.uniform(0.2, 0.5) * direction
                
                pts_end[:, 0] += shift # 整体横移 (模拟扫视的主要分量)
                
                # 透视挤压 (Perspective Squeeze)
                squeeze = magnitude * direction
                
                # 左边两个点 (0, 3) 上下收缩/扩张
                pts_end[0, 1] += squeeze # y 增加 (下移)
                pts_end[3, 1] -= squeeze # y 减小 (上移)
                # 右边两个点 (1, 2) 上下扩张/收缩
                pts_end[1, 1] -= squeeze # y 减小
                pts_end[2, 1] += squeeze # y 增加

            elif rot_type == 'pitch':
                # --- [新增] 模拟点头 ---
                
                # [修改] 限制最大畸变幅度
                max_squeeze = w * 0.2
                magnitude = np.random.uniform(w * 0.05, max_squeeze)
                
                direction = np.random.choice([-1, 1])
                
                shift = h * np.random.uniform(0.2, 0.5) * direction
                pts_end[:, 1] += shift # 整体纵移
                
                # 上下透视挤压
                # 注意 Pitch 时是对 x 坐标操作
                squeeze = magnitude * direction
                
                pts_end[0, 0] += squeeze
                pts_end[1, 0] -= squeeze
                pts_end[3, 0] -= squeeze
                pts_end[2, 0] += squeeze

            # === 通用漂移 (你原本的 Drift) ===
            max_drift = min(W_raw, H_raw) * 0.05
            pts_end[:, 0] += np.random.uniform(-max_drift, max_drift)
            pts_end[:, 1] += np.random.uniform(-max_drift, max_drift)
            
        else:
            # === 默认混合模式 (Mixed) ===
            # 1. 定义起始视窗
            scale_start = np.random.uniform(0.5, 0.7) 
            w_s, h_s = int(W_raw * scale_start), int(H_raw * scale_start)
            x_s = np.random.randint(0, W_raw - w_s)
            y_s = np.random.randint(0, H_raw - h_s)
            
            pts_start = np.float32([
                [x_s, y_s], 
                [x_s + w_s, y_s], 
                [x_s + w_s, y_s + h_s], 
                [x_s, y_s + h_s]
            ])
            center_x_s = x_s + w_s / 2
            center_y_s = y_s + h_s / 2

            # 2. 定义结束视窗
            valid_end_found = False
            pts_end = None
            
            for _ in range(10):
                if np.random.rand() < 0.5:
                    scale_end = np.random.uniform(0.4, min(scale_start - 0.1, 0.6))
                else:
                    scale_end = np.random.uniform(max(scale_start + 0.1, 0.6), 0.8)
                    
                w_e, h_e = int(W_raw * scale_end), int(H_raw * scale_end)
                base_dim = min(W_raw, H_raw)
                min_shift = base_dim * 0.30
                max_shift = base_dim * 0.60
                
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(min_shift, max_shift)
                
                shift_x = int(dist * np.cos(angle))
                shift_y = int(dist * np.sin(angle))
                
                center_x_e = center_x_s + shift_x
                center_y_e = center_y_s + shift_y
                
                x_e = int(center_x_e - w_e / 2)
                y_e = int(center_y_e - h_e / 2)
                
                x_e = np.clip(x_e, 0, W_raw - w_e)
                y_e = np.clip(y_e, 0, H_raw - h_e)
                
                real_dist = np.sqrt((x_e + w_e/2 - center_x_s)**2 + (y_e + h_e/2 - center_y_s)**2)
                
                if real_dist > min_shift * 0.5:
                    valid_end_found = True
                    break
            
            if not valid_end_found:
                w_e, h_e = int(W_raw * scale_end), int(H_raw * scale_end)
                x_e = np.random.randint(0, W_raw - w_e)
                y_e = np.random.randint(0, H_raw - h_e)

            pts_end = np.float32([
                [x_e, y_e], 
                [x_e + w_e, y_e], 
                [x_e + w_e, y_e + h_e], 
                [x_e, y_e + h_e]
            ])
            
            # 2.1 显式加入平面内旋转
            rot_angle = np.random.uniform(-45, 45)
            cx, cy = np.mean(pts_end[:, 0]), np.mean(pts_end[:, 1])
            rot_mat = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
            pts_end = cv2.transform(np.array([pts_end]), rot_mat)[0]

            # 3. 加入透视畸变
            distortion_magnitude = min(w_e, h_e) * 0.2
            noise = np.random.uniform(-distortion_magnitude, distortion_magnitude, size=pts_end.shape)
            pts_end += noise

            # 边界保护
            pts_end[:, 0] = np.clip(pts_end[:, 0], 0, W_raw - 1)
            pts_end[:, 1] = np.clip(pts_end[:, 1], 0, H_raw - 1)
        
        return pts_start, pts_end

    def render_sequence(self, img, pts_start, pts_end, video_len=None):
        """
        核心渲染层：给定起始点和结束点，渲染图像序列。
        如果你想手动控制变换，直接调用这个函数传入自定义的 pts_start/pts_end 即可。
        """
        if video_len is None:
            video_len = np.random.randint(self.min_frames, self.max_frames + 1)
            
        dst_pts_canvas = np.float32([
            [0, 0], 
            [self.width, 0], 
            [self.width, self.height], 
            [0, self.height]
        ])
        
        images = []
        timestamps = []
        
        for i in range(video_len):
            alpha = i / (video_len - 1)
            # 使用 smooth step 插值让运动更自然
            t = alpha * alpha * (3 - 2 * alpha)
            
            curr_src_pts = pts_start * (1 - t) + pts_end * t
            M = cv2.getPerspectiveTransform(curr_src_pts, dst_pts_canvas)
            
            warped_frame = cv2.warpPerspective(img, M, (self.width, self.height), flags=cv2.INTER_LINEAR)
            images.append(warped_frame)
            timestamps.append(i / self.fps)

        images = np.array(images) 
        tensor_imgs = torch.from_numpy(images).float() / 255.0
        tensor_imgs = tensor_imgs.permute(1, 2, 0)
        tensor_ts = torch.tensor(timestamps).float()
        tensor_ts = tensor_ts * self.dt 
        
        return {
            'images': tensor_imgs,
            'timestamps': tensor_ts,
            'video_len': torch.tensor(video_len)
        }

    def get_video_sequence(self, image_path):
        # 流程整合层
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        H_raw, W_raw = img.shape
        
        # 1. 生成轨迹参数 (可被重载或跳过)
        pts_start, pts_end = self.sample_trajectory_params(H_raw, W_raw)
        
        # 2. 渲染序列
        return self.render_sequence(img, pts_start, pts_end)

class COCODatasetIterator(Dataset):
    def __init__(self, folder, height, width, min_frames, max_frames, fps=200, mode='mixed'):
        if os.path.isfile(folder) and folder.endswith('.txt'):
            with open(folder, 'r') as f:
                self.files = [line.strip() for line in f if line.strip()]
            print(f"[Dataset] Loaded {len(self.files)} images from list file: {folder}")
        else:
            self.files = sorted(glob.glob(os.path.join(folder, "*.jpg"))) + sorted(glob.glob(os.path.join(folder, "*.png")))
            print(f"[Dataset] Scanned {len(self.files)} images from folder: {folder}")
            
        self.streamer = SyntheticMotionStream(height, width, min_frames, max_frames, fps=fps, mode=mode)
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        for i in range(20):
            curr_idx = (idx + i) % len(self.files)
            data = self.streamer.get_video_sequence(self.files[curr_idx])
            if data is not None:
                data['first_times'] = torch.tensor(1) 
                return data
        raise RuntimeError("Failed to load valid images from the list. Please check file paths.")

def make_dataloader(folder, batch_size, height, width, min_frames, max_frames, fps=200, mode='mixed'):
    dataset = COCODatasetIterator(folder, height, width, min_frames, max_frames, fps=fps, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)