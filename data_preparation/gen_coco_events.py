import sys
import os
import torch
import numpy as np
import argparse
import cv2
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation.dataset_syn.esim import SyntheticEventSimulator
from data_preparation.dataset_syn.stream_dataset import make_dataloader
from data_preparation.dataset_syn.my_gpu_simulator import GPUEventSimulator 

class SimpleSimulatorConfig:
    @classmethod
    def from_params(cls, folder, batch_size, height, width, min_frames, max_frames, fps, device='cuda:0', mode='mixed' ):
        dataloader = make_dataloader(folder, batch_size, height, width, min_frames, max_frames, fps=fps, mode=mode)
        event_gpu = GPUEventSimulator(batch_size, height, width)
        event_gpu.to(device)
        return SyntheticEventSimulator(dataloader, event_gpu, device)

class SyntheticEventSimulator(object):
    def __init__(self, dataloader, simulator, device):
        self.dataloader = dataloader
        self.simulator = simulator
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            gray_images = batch['images'].to(self.device) 
            timestamps = batch['timestamps'].to(self.device)
            video_len = batch['video_len'].to(self.device)
            first_times = batch['first_times'].to(self.device)
            
            if gray_images.shape[0] == 1:
                gray_images = gray_images.squeeze(0)

            log_images = self.simulator.dynamic_moving_average(gray_images, video_len, timestamps, first_times)
            raw_events = self.simulator.get_events(log_images, video_len, timestamps, first_times)
            
            yield {
                'raw_events': raw_events,
                'images': gray_images,
                'timestamps': timestamps
            }

def generate(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    sim = SimpleSimulatorConfig.from_params(
        folder=args.coco_path,
        batch_size=1, 
        height=args.height,
        width=args.width,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        fps=args.fps,
        device=args.device,
        mode=args.mode
    )
    
    print(f"Generating {args.num_samples} sequences...")
    print(f"Structure: seq_xxx/ -> [events/events.npz, frames/*.png, frame_time.txt]")
    
    count = 0
    with torch.no_grad():
        for batch in tqdm(sim):
            if count >= args.num_samples:
                break
            
            # === 创建目录结构 ===
            seq_root = os.path.join(args.output_dir, f'seq_{count:06d}')
            os.makedirs(seq_root, exist_ok=True)
            
            events_dir = os.path.join(seq_root, 'events')
            frames_dir = os.path.join(seq_root, 'frames')
            os.makedirs(events_dir, exist_ok=True)
            os.makedirs(frames_dir, exist_ok=True)
            
            # 1. 保存 events
            # events = batch['raw_events'].cpu().numpy()  <-- 删除或注释掉这两行旧代码
            
            # === [修改] 在 GPU 上直接排序 ===
            raw_events = batch['raw_events']  # 此时还在 GPU 上
            
            # 检查是否有事件
            if raw_events.shape[0] > 0:
                # 按照第4列 (时间戳) 进行排序
                # torch.argsort 返回排序后的索引
                # raw_events[:, 4] 是时间列
                sorted_indices = torch.argsort(raw_events[:, 4])
                
                # 使用索引重排事件 (仍在 GPU 上)
                raw_events = raw_events[sorted_indices]

            # 排序后再转到 CPU 并转为 numpy
            events = raw_events.cpu().numpy()

            np.savez_compressed(os.path.join(events_dir, 'events.npz'), events=events)

            # 2. 保存 frames 和 frame_time.txt
            imgs = batch['images'].cpu().numpy() # [H, W, T]
            ts = batch['timestamps'].cpu().numpy() # [B, T] or [T]
            if ts.ndim == 2: ts = ts[0]
            
            # [H, W, T] -> [T, H, W]
            imgs = np.transpose(imgs, (2, 0, 1))
            
            time_file_path = os.path.join(seq_root, 'frame_time.txt')
            with open(time_file_path, 'w') as f:
                # 写入头部 (可选，这里不写，只写数据)
                # f.write("filename timestamp\n")
                
                for i, (frame, t) in enumerate(zip(imgs, ts)):
                    frame_name = f'frame_{i:04d}.png'
                    
                    # 保存图像
                    if frame.max() <= 1.0: frame = frame * 255
                    cv2.imwrite(os.path.join(frames_dir, frame_name), frame.astype(np.uint8))
                    
                    # 写入时间戳
                    # 格式: frame_0000.png 0.005000
                    f.write(f'{frame_name} {t:.6f}\n')
            
            count += 1
            
    print(f"Done. Saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', type=str, default="dataset/selected_coco_planar_organic.txt")
    # parser.add_argument('--coco_path', type=str, default="data/coco/train")
    parser.add_argument('--output_dir', type=str, default="dataset/gen_coco_1")
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--height', type=int, default=260)
    parser.add_argument('--width', type=int, default=346)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--min_frames', type=int, default=72)
    parser.add_argument('--max_frames', type=int, default=120)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--mode', type=str, default='mixed', choices=['mixed', 'rotation'])

    args = parser.parse_args()
    generate(args)