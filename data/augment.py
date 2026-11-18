import os 
from typing import Any, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import albumentations as A
import cv2
import random

def modality_dropout_weighted(frame, event, probabilities=[0.7, 0.15, 0.15]):
    """
    根据指定的概率，在三个场景中选择一个来执行：不丢弃、丢弃模态A、丢弃模态B。

    Args:
        modality1 (np.ndarray): 模态A的图像。
        modality2 (np.ndarray): 模态B的图像。
        probabilities (list): 一个包含三个浮点数的列表，分别代表：
                              [P(不丢弃), P(丢弃A), P(丢弃B)]
                              列表元素之和必须为 1.0。
    """
    scenarios = ["none", "dropout_F", "dropout_E"]
    chosen_scenario = random.choices(scenarios, weights=probabilities, k=1)[0]

    # 根据选择的场景执行操作
    if chosen_scenario == "dropout_F":
        frame = np.ones_like(frame)*255 if random.random() < 0.5 else np.zeros_like(frame)
    elif chosen_scenario == "dropout_E":
        event = np.zeros_like(event)
    # else: chosen_scenario == "none"

    return frame, event

def random_time_scale(
    events: Dict[str, np.ndarray], scale: Tuple[float, float] = (0.8, 1.2)
) -> Dict[str, np.ndarray]:
    """
    Randomly scale the time of events.
    """
    if scale is not None:
        t_start = events["t"][..., 0]
        events["t"] = (events["t"] - t_start) * np.random.uniform(
            scale[0], scale[1]
        ) + t_start
    return events


def slice_events(
    events: Dict[str, np.ndarray], slice_dt: Tuple[float, float], mode: str
) -> Dict[str, np.ndarray]:
    """
    Slice the events into a time crop with the given time interval dt(milliseconds).
    """
    if slice_dt is not None:
        dt = np.random.uniform(slice_dt[0], slice_dt[1])
        t = events["t"]
        t_length = t[..., -1] - t[..., 0]  # microseconds
        if t_length > dt * 1000.0:
            if mode == "start":
                t_start = t[..., 0]
                t_end = t_start + dt / 1000.0
                t_mask = (t >= t_start) & (t < t_end)
                events = {k: v[t_mask] for k, v in events.items()}
            elif mode == "end":
                t_end = t[..., -1]
                t_start = t_end - dt / 1000.0
                t_mask = (t >= t_start) & (t < t_end)
                events = {k: v[t_mask] for k, v in events.items()}
            elif mode == "middle":
                t_middle = (t[..., 0] + t[..., -1]) / 2
                t_start = t_middle - dt / 2000.0
                t_end = t_middle + dt / 2000.0
                t_mask = (t >= t_start) & (t < t_end)
                events = {k: v[t_mask] for k, v in events.items()}
            elif mode == "random":
                t_start = t[..., 0]
                t_end = t[..., -1] - dt / 1000.0
                t_random = np.random.uniform(t_start, t_end)
                t_mask = (t >= t_random) & (t < t_random + dt / 1000.0)
                events = {k: v[t_mask] for k, v in events.items()}
            else:
                raise ValueError(f"Unknown mode: {mode}")

    return events


def random_time_flip(
    events: Dict[str, np.ndarray], p: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Randomly flip the time of events.
    """
    if np.random.rand() < p:
        events["t"] = events["t"][::-1]
        events["t"] = events["t"][0] - events["t"]
        events["p"] = -events["p"].astype(
            np.float32
        )  # Inversion in time means inversion in polarity

    return events


def add_correlated_events(
    event: Dict[str, np.ndarray],
    xy_std: float = 1.5,
    ts_std: float = 0.1,
    percent: Tuple[float, float] = (0.001, 0.01),
) -> Dict[str, np.ndarray]:
    """
    Add correlated events to the event.

    Args:
        event (dict): Dictionary with keys "x", "y", "t", "p" and values numpy arrays of the same length.
        xy_std (float): Standard deviation of the gaussian distribution for the x and y coordinates.
        ts_std (float): Standard deviation of the gaussian distribution for the timestamp.

    Returns:
        event (dict): Dictionary with keys "x", "y", "t", "p" and values numpy arrays of the same length.
    """

    N = event["x"].shape[0]
    if N < 1000:
        return event
    to_add = np.random.randint(int(N * percent[0]), int(N * percent[1]))
    event_new = {
        "x": event["x"] + np.random.normal(0, xy_std, size=event["x"].shape),
        "y": event["y"] + np.random.normal(0, xy_std, size=event["x"].shape),
        "t": event["t"] + np.random.normal(0, ts_std / 1000.0, size=event["x"].shape),
        "p": event["p"],
    }
    idx = np.random.choice(np.arange(N), size=to_add, replace=False)
    for key in event_new:
        event_new[key] = event_new[key][idx]
    event_new["x"] = np.clip(event_new["x"], 0, event["x"].max())
    event_new["y"] = np.clip(event_new["y"], 0, event["y"].max())
    for key in event:
        event[key] = np.concatenate((event[key], event_new[key]))
    idx_sorted = np.argsort(event["t"], axis=0)[::-1]
    for key in event:
        event[key] = event[key][idx_sorted]
    return event


def random_delete_events(
    event: Dict[str, np.ndarray], percent: Tuple[float, float] = (0.001, 0.01)
) -> Dict[str, np.ndarray]:
    """
    Randomly delete events.
    """
    N = event["x"].shape[0]
    if N > 1000:
        to_delete = np.random.randint(int(N * percent[0]), int(N * percent[1]))
        idx = np.random.choice(np.arange(N), size=N - to_delete, replace=False)
        for key in event:
            event[key] = event[key][idx]
    return event


def random_crop_pair(
    x: torch.Tensor, y: torch.Tensor, crop_size: Tuple[int, int], mask=None
):

    assert x.shape[-2:] == y.shape[-2:], f"Shape mismatch: {x.shape} and {y.shape}"
    assert (
        x.shape[-2] >= crop_size[0] and x.shape[-1] >= crop_size[1]
    ), f"Shape mismatch: {x.shape} and {crop_size}"

    H, W = x.shape[-2:]
    h, w = crop_size
    x0 = np.random.randint(0, H - h)
    y0 = np.random.randint(0, W - w)
    x = x[..., x0 : x0 + h, y0 : y0 + w]
    y = y[..., x0 : x0 + h, y0 : y0 + w]
    if mask is not None:
        mask = mask[..., x0 : x0 + h, y0 : y0 + w]
        return x, y, mask

    return x, y


def random_flip_pair(
    x: torch.Tensor, y: torch.Tensor, p_h: float = 0.5, p_w: float = 0.5, mask=None
):
    if np.random.rand() < p_h:
        x = torch.flip(x, dims=(-2,))
        y = torch.flip(y, dims=(-2,))
        if mask is not None:
            mask = torch.flip(mask, dims=(-2,))
    if np.random.rand() < p_w:
        x = torch.flip(x, dims=(-1,))
        y = torch.flip(y, dims=(-1,))
        if mask is not None:
            mask = torch.flip(mask, dims=(-1,))

    if mask is not None:
        return x, y, mask
    else:
        return x, y


def random_rotate_pair(
    x: torch.Tensor, y: torch.Tensor, angle: float = 10.0, mask=None
):
    angle = np.random.uniform(-angle, angle)
    x = TF.rotate(x, angle)
    y = TF.rotate(y, angle)
    if mask is not None:
        mask = TF.rotate(mask, angle)
        return x, y, mask
    return x, y


class EventPointsAugmentation:

    def __init__(
        self,
        time_scale: Tuple[float, float] = (0.5, 2.0),
        slice_dt: Tuple[float, float] = (30, 100),
        slice_mode: str = "random",
        flip_p: float = 0.5,
        xy_std: float = 1.5,
        ts_std: float = 0.1,
        add_percent: Tuple[float, float] = (0.01, 0.001),
        del_percent: Tuple[float, float] = (0.01, 0.001),
    ) -> None:
        self.time_scale = time_scale
        self.slice_dt = slice_dt
        self.slice_mode = slice_mode
        self.flip_p = flip_p
        self.xy_std = xy_std
        self.ts_std = ts_std
        self.add_percent = add_percent
        self.del_percent = del_percent

    def __call__(self, events) -> Any:
        # events = random_time_scale(events, scale=self.time_scale)
        # events = slice_events(events, slice_dt=self.slice_dt, mode=self.slice_mode)
        # events = random_time_flip(events, p=self.flip_p)
        # events = add_correlated_events(events, self.xy_std, self.ts_std, self.add_percent)
        # events = random_delete_events(events, self.del_percent)
        return events


class PairAugmentation:
    def __init__(
        self,
        crop_size: Tuple[int, int],
        flip_p_h: float = 0.5,
        flip_p_w: float = 0.5,
        rotate_angle: float = 10.0,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.flip_p_h = flip_p_h
        self.flip_p_w = flip_p_w
        self.rotate_angle = rotate_angle

    def __call__(self, x: torch.Tensor, y: torch.Tensor, mask=None) -> Any:
        x, y, mask = random_flip_pair(
            x, y, p_h=self.flip_p_h, p_w=self.flip_p_w, mask=mask
        )
        x, y, mask = random_rotate_pair(x, y, angle=self.rotate_angle, mask=mask)
        x, y, mask = random_crop_pair(x, y, crop_size=self.crop_size, mask=mask)

        return x, y, mask


class ImageArrayAugmentation:
    def __init__(
        self,
        gamma_p: float = 0.1,
        gamma_limit: Tuple[int, int] = (15, 65),
        brightness_p: float = 0.5,
        brightness_limit: Tuple[float, float] = (-0.3, -0.0),
        contrast_limit: Tuple[float, float] = (-0.5, 0.3),
        noise_p: float = 0.5,
        motion_blur_p: float = 0,
        motion_blur_limit: Tuple[int, int] = (5, 35),
        dropout_p: float = 0.5,
        dropout_size_limit: Tuple[int, int] = (0.1, 0.2),
        dropout_num_limit: Tuple[int, int] = (1, 8),
        dropout_fill = 'random_uniform',
        hdr_p: float = 0.5,
        # full_dropout_p = 0.0,
    ) -> None:
        self.gamma_p = gamma_p
        self.gamma_limit = gamma_limit
        self.brightness_p = brightness_p
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.noise_p = noise_p
        self.motion_blur_p = motion_blur_p
        self.dropout_p = dropout_p
        self.hdr_p = hdr_p

        self.transform = A.Compose(
            [   
                HDRIllumination(p=self.hdr_p),
                A.RandomGamma(p=self.gamma_p, gamma_limit=self.gamma_limit),
                A.RandomBrightnessContrast(
                    p=self.brightness_p,
                    brightness_limit=self.brightness_limit,
                    contrast_limit=self.contrast_limit,
                ),
                A.GaussNoise(p=self.noise_p),
                A.MotionBlur(p=motion_blur_p, blur_limit=motion_blur_limit),
                A.CoarseDropout(
                    num_holes_range = dropout_num_limit,
                    hole_height_range = dropout_size_limit,
                    hole_width_range = dropout_size_limit,
                    fill=dropout_fill,
                    fill_mask=0,
                    p=dropout_p  
                ),
                # ModalityDropout(p=full_dropout_p)
            ]
        )

    def __call__(self, img: np.ndarray) -> Any:
        mask = np.ones_like(img)
        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        return img, mask

class GeometricAugmentation:
    """
    对图像和事件表示应用相同的、同步的几何（空间）变换。
    """
    def __init__(
        self,
        rotate_p: float = 1,
        rotate_limit: int = 45,
        flip_p: float = 0.5,
        shift_scale_rotate_p: float = 0.5,
    ) -> None:
        # 定义一个包含所有几何变换的 Compose 流程
        self.transform = A.Compose(
            [   
                A.Rotate(p=rotate_p, limit=rotate_limit, border_mode=cv2.BORDER_REFLECT101),
                # A.HorizontalFlip(p=flip_p),
                A.ShiftScaleRotate(
                    p=shift_scale_rotate_p,
                    shift_limit=0.05, 
                    scale_limit=0.1, 
                    rotate_limit=0, # Rotate 已单独定义，这里设为0
                    border_mode=cv2.BORDER_REFLECT101,
                ),
            ],
            # 关键：定义 'events' 作为一个需要同步变换的额外目标
            additional_targets={'events': 'image',
                                'image_raw': 'image'},
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        )

    def __call__(self, image: np.ndarray, events: np.ndarray, image_raw: np.ndarray,
                  matched_kpts: np.ndarray, keypoints_map: np.ndarray) -> Dict[str, np.ndarray]:
        """
        对输入的 image 和 events 应用变换。

        Args:
            image (np.ndarray): 图像数组，格式为 (H, W, C)。
            events (np.ndarray): 事件表示数组，格式为 (H, W, C)。

        Returns:
            Dict[str, np.ndarray]: 一个包含变换后 'image' 和 'events' 的字典。
        """
        # 同时传入 image 和 events，Albumentations 会确保变换参数一致
        return self.transform(image=image, events=events, image_raw=image_raw, keypoints=matched_kpts, mask = keypoints_map)

class ModalityDropout(A.ImageOnlyTransform):
    """
    以一定概率丢弃图像（填充为常数值）。
    用于模拟传感器失效或模态缺失。
    """
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # 创建全填充的图像
        color = np.random.randn()
        return np.full_like(img, color*255)

    def get_transform_init_args_names(self):
        return ("fill_value",)


class HDRIllumination(A.ImageOnlyTransform):
    """
    自定义的 HDR 光照增强变换，包含：全局低光、空间阴影、过曝。
    """
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # 1. 确保转为 float32 [0, 1] 进行计算
        is_uint8 = (img.dtype == np.uint8)
        img_f = img.astype(np.float32) / 255.0 if is_uint8 else img.copy()
        
        # 2. 三选一模式
        # 调整概率：稍微增加了 Mode 1 (阴影) 的权重，因为它对 HDR 训练最重要
        mode = np.random.choice([0, 1, 2], p=[0.3, 0.6, 0.1])

        # === Mode 0: Global Low-Light (全局暗) ===
        if mode == 0:
            gamma = np.random.uniform(3.0, 6.0)
            gain = np.random.uniform(0.05, 0.2)
            noise_std = np.random.uniform(0.02, 0.05)
            
            img_f = img_f ** gamma
            img_f = img_f * gain
            noise = np.random.normal(0, noise_std, img_f.shape)
            img_f = img_f + noise

        # === Mode 1: Spatial Shadow (强对比阴影) ===
        elif mode == 1:
            H, W = img_f.shape[:2]
            x = np.linspace(0, 1, W)
            y = np.linspace(0, 1, H)
            grid_x, grid_y = np.meshgrid(x, y)
            
            # --- [修改核心] 使用 Sigmoid 制造清晰的明暗分界线 ---
            
            # 1. 随机定义一条分界线 (Line)
            # 通过一个中心点 (cx, cy) 和一个角度 angle 定义
            cx = np.random.uniform(0.2, 0.8) # 阴影边缘的中心位置 (0.5 为正中间)
            cy = np.random.uniform(0.2, 0.8)
            angle = np.random.uniform(0, 2 * np.pi)
            
            # 计算法向量
            normal_x = np.cos(angle)
            normal_y = np.sin(angle)
            
            # 2. 计算每个像素到分界线的距离 (有正负)
            # dist > 0 为亮区，dist < 0 为暗区 (或者反过来)
            dist_map = (grid_x - cx) * normal_x + (grid_y - cy) * normal_y
            
            # 3. 使用 Sigmoid 函数控制边缘硬度 (Sharpness)
            # sharpness 越大，边缘越硬（对比越明显）；越小越模糊
            sharpness = np.random.uniform(10, 20) 
            shadow_map = 1 / (1 + np.exp(-sharpness * dist_map))
            
            # 4. 随机反转明暗方向
            if np.random.rand() < 0.5:
                shadow_map = 1 - shadow_map
                
            # 5. 调整亮度范围
            # min_light 设得更低 (0.01~0.15)，模拟深邃的阴影
            min_light = np.random.uniform(0.00, 0.05) 
            # max_light 稍微随机一点，不一定非要全是 1.0
            max_light = np.random.uniform(1.1, 1.5)
            
            shadow_map = shadow_map * (max_light - min_light) + min_light
            
            # 扩展维度以匹配通道
            if len(img_f.shape) == 3 and img_f.shape[2] > 1:
                shadow_map = shadow_map[..., None]

            # 6. 应用阴影
            img_f = img_f * shadow_map
            
            # 7. 在暗部添加更强的噪声 (模拟 Sensor 在低光下的表现)
            # 噪声强度与亮度成反比：越暗噪声越大
            noise_intensity = np.random.uniform(0.02, 0.05)
            noise = np.random.normal(0, noise_intensity, img_f.shape) * (1 - shadow_map)
            img_f = img_f + noise

        # === Mode 2: Over-Exposure (过曝) ===
        elif mode == 2:
            gain = np.random.uniform(1.2, 2.5)
            img_f = img_f * gain

        # 3. 还原回 uint8
        img_final = np.clip(img_f, 0, 1.0) * 255.0
        
        if is_uint8:
            return img_final.astype(np.uint8)
        else:
            return (img_final / 255.0).astype(np.float32)

if __name__ == "__main__":
    pass


