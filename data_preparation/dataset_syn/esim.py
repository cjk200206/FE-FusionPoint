import torch
import numpy as np
from .stream_dataset import make_dataloader
from .my_gpu_simulator import GPUEventSimulator 

# 注意：这里我们复用原项目中的 GPUEventSimulator，因为它涉及 CUDA 编译较复杂，
# 只要路径对就行。如果不行，你需要把 my_gpu_simulator.py 也复制到 dataset_syn 下。

class SyntheticEventSimulator(object):
    def __init__(self, dataloader, simulator, device):
        self.dataloader = dataloader
        self.simulator = simulator
        self.device = device
        
    @classmethod
    def from_params(cls, folder, batch_size, height, width, min_frames=20, max_frames=30, device='cuda:0'):
        dataloader = make_dataloader(folder, batch_size, height, width, min_frames, max_frames)
        # 初始化 GPU 模拟器
        event_gpu = GPUEventSimulator(batch_size, height, width)
        event_gpu.to(device)
        return cls(dataloader, event_gpu, device)

    def __iter__(self):
        for batch in self.dataloader:
            # 1. 准备数据
            # [B, H, W, T] -> [B, H, W, T] (images)
            gray_images = batch['images'].to(self.device) 
            timestamps = batch['timestamps'].to(self.device) # [B, T]
            video_len = batch['video_len'].to(self.device)
            first_times = batch['first_times'].to(self.device)
            
            # Simulator 需要 images 是 [H, W, T] (如果 batch=1) 或者 [B, H, W, T] ??
            # 检查原代码：gray_images = batch['images'].squeeze(0) -> [H, W, T] (当batch=1时)
            # 如果 batch > 1，原代码的 dynamic_moving_average 可能需要调整。
            # 为简单起见，且通常 batch=1 生成数据：
            
            if gray_images.shape[0] == 1:
                gray_images = gray_images.squeeze(0) # [H, W, T]
            else:
                 # GPUEventSimulator usually expects [B, H, W, T] or handles batch inside?
                 # Looking at my_gpu_simulator.py (not provided but inferred), 
                 # usually it expects batched tensors if initialized with batch_size > 1.
                 # Let's align with original: squeeze if batch=1 seems to imply sim handles H,W,T for single batch?
                 # NO, original code: gray_images = batch['images'].squeeze(0) implies batch_size=1
                 pass

            # 2. 生成对数图像插值 (Log Images)
            log_images = self.simulator.dynamic_moving_average(gray_images, video_len, timestamps, first_times)

            # 3. 生成事件
            # [N, 4] or [N, 5]
            raw_events = self.simulator.get_events(log_images, video_len, timestamps, first_times)
            
            # 4. 构造输出
            out_batch = {
                'raw_events': raw_events,
                'images': gray_images, # 原始灰度图序列
                'timestamps': timestamps,
                'video_len': video_len
            }
            
            yield out_batch