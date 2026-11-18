# Copyright (c) Prophesee S.A.
# Modified for FE-FusionPoints; see THIRD_PARTY_LICENSES.txt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
More efficient reimplementation.
The main difference is cuda kernels & possibility to
directly stream the voxel grid.
"""
# pylint: disable=access-member-before-definition
# pylint: disable=undefined-variable
# pylint: disable=not-callable

import torch
import torch.nn as nn
import numpy as np

from numba import cuda

from metavision_core_ml.video_to_event.cutoff_kernels import _cuda_kernel_dynamic_moving_average, _cpu_kernel_dynamic_moving_average
from .my_events_kernels import _cuda_kernel_count_events, _cpu_kernel_count_events
from .my_events_kernels import _cpu_kernel_fill_events, _cuda_kernel_fill_events
from .my_events_kernels import _cuda_kernel_time_surface_sequence, _cpu_kernel_time_surface_sequence
from .my_events_kernels import _cuda_kernel_time_surface3D_sequence, _cpu_kernel_time_surface3D_sequence
from .my_events_kernels import _cuda_kernel_voxel_grid_and_ts_sequence, _cpu_kernel_voxel_grid_and_ts_sequence
from .my_events_kernels import _cuda_kernel_voxel_grid_sequence, _cpu_kernel_voxel_grid_sequence
from .my_events_kernels import _cuda_kernel_voxel_grid_DA_sequence, _cpu_kernel_voxel_grid_DA_sequence
from .my_events_kernels import _cuda_kernel_voxel_grid_sequence_5, _cpu_kernel_voxel_grid_sequence_5
from .my_events_kernels import _cpu_kernel_voxel_grid_TenCode_sequence, _cuda_kernel_voxel_grid_TenCode_sequence

class GPUEventSimulator(nn.Module):
    """
    GPU Event Simulator of events from frames & timestamps.

    Implementation is based on the following publications:

    - Video to Events: Recycling Video Datasets for Event Cameras: Daniel Gehrig et al.
    - V2E: From video frames to realistic DVS event camera streams: Tobi Delbruck et al.

    Args:
        batch_size (int): number of video clips / batch
        height (int): height
        width (int): width
        c_mu (float or list): threshold average
                              if scalar will consider same OFF and ON thresholds
                              if list, will be considered as [ths_OFF, ths_ON] 
        c_std (float): threshold standard deviation
        refractory period (int): time before event can be triggered again
        leak_rate_hz (float): frequency of reference voltage leakage
        cutoff_hz (float): frequency for photodiode latency
        shot_noise_hz (float): frequency of shot noise events
    """

    def __init__(
            self, batch_size, height, width, c_mu=0.2, c_std=0, refractory_period=10, leak_rate_hz=0, cutoff_hz=0,
            shot_noise_hz=0):
        super().__init__()
        shape1 = (batch_size, height, width)
        shape2 = (2, batch_size, height, width)
        self.register_buffer("log_states", torch.zeros(shape1, dtype=torch.float64))
        self.register_buffer("prev_log_images", torch.zeros(shape1, dtype=torch.float32))
        self.register_buffer("counts", torch.zeros(shape1, dtype=torch.int32))
        self.register_buffer("timestamps", torch.zeros(shape1, dtype=torch.float32))

        if isinstance(c_mu, list):
            assert len(c_mu) >= 2
            assert c_mu[0] > 0, "contrast thresholds must be positive"
            assert c_mu[1] > 0, "contrast thresholds must be positive"
            self.threshold_mu = c_mu
        else:
            assert c_mu > 0, "contrast thresholds must be positive"
            self.threshold_mu = [c_mu, c_mu]

        thresholds = torch.randn(shape2, dtype=torch.float32)
        assert c_std >= 0, "contrast thresholds std deviation must be non-negative"
        self.threshold_std = c_std
        for p in [0, 1]:
            thresholds[p, :] = thresholds[p, :]*c_std + self.threshold_mu[p]
        self.register_buffer("thresholds", thresholds)
        self.register_buffer("prev_image_ts", torch.zeros((batch_size), dtype=torch.float32))
        self.register_buffer("filtering_prev_image_ts", torch.zeros((batch_size), dtype=torch.float32))
        self.thresholds.clamp_(0.01, 1.0)

        self.register_buffer("prev_log_images_0", torch.zeros(shape1, dtype=torch.float32))
        self.register_buffer("prev_log_images_1", torch.zeros(shape1, dtype=torch.float32))

        self.register_buffer("rng_states", torch.zeros(shape1, dtype=torch.float32))

        # generate array of noise parameters
        self.register_buffer("refractory_periods", torch.LongTensor([refractory_period] * batch_size))
        self.register_buffer("cutoff_rates", torch.tensor([cutoff_hz] * batch_size, dtype=torch.float32))
        self.register_buffer("leak_rates", torch.tensor([leak_rate_hz * 1e-6] * batch_size, dtype=torch.float32))
        self.register_buffer("shot_rates", torch.tensor([shot_noise_hz * 1e-6] * batch_size, dtype=torch.float32))
        self.register_buffer("threshold_mus", torch.tensor(self.threshold_mu * batch_size,
                                                           dtype=torch.float32).reshape(-1, 2))  # shape (B,2)

    def get_size(self):
        return self.thresholds.shape[-2:]

    def randomize_broken_pixels(self, first_times, video_proba=1e-2, crazy_pixel_proba=5e-4, dead_pixel_proba=5e-3):
        """
        Simulates dead & crazy pixels

        Args:
            first_times: B video just started flags
            video_proba: probability to simulate broken pixels
        """
        height, width = self.thresholds.shape[-2:]
        device = first_times.device
        for i in range(len(first_times)):
            if first_times[i].item():
                # crazy pixels
                if np.random.rand() < video_proba:
                    mask = torch.rand(height, width).to(device) < crazy_pixel_proba
                    self.thresholds[:, i, mask] = 1e-3
                # dead pixels
                if np.random.rand() < video_proba:
                    mask = torch.rand(height, width).to(device) < dead_pixel_proba
                    self.thresholds[:, i, mask] = 1e3

    def randomize_thresholds(self, first_times, th_mu_min=0.05, th_mu_max=0.2, th_std_min=0.001, th_std_max=0.01):
        """
        Re-Randomizes thresholds per video

        Args:
            first_times: B video just started flags
            th_mu_min (scalar or list of scalars): min average threshold 
                              if list, will be considered as [th_mu_min_OFF, th_mu_min_ON] 
            th_mu_max (scalar or list of scalars): max average threshold 
                              if list, will be considered as [th_mu_max_OFF, th_mu_max_ON] 
            th_std_min: min threshold standard deviation
            th_std_max: max threshold standard deviation
        """

        # support both symmetric and asymmetric OFF and ON thresholds
        if not isinstance(th_mu_min, list):
            assert th_mu_min > 0, "contrast thresholds must be positive"
            th_mu_min_list = [th_mu_min, th_mu_min]
        else:
            assert len(th_mu_min) >= 2
            assert th_mu_min[0] > 0, "contrast thresholds must be positive"
            assert th_mu_min[1] > 0, "contrast thresholds must be positive"
            th_mu_min_list = th_mu_min

        if not isinstance(th_mu_max, list):
            assert th_mu_max > 0, "contrast thresholds must be positive"
            th_mu_max_list = [th_mu_max, th_mu_max]
        else:
            assert len(th_mu_max) >= 2
            assert th_mu_max[0] > 0, "contrast thresholds must be positive"
            assert th_mu_max[1] > 0, "contrast thresholds must be positive"
            th_mu_max_list = th_mu_max

        batch_size = len(first_times)
        ft = first_times[:, None, None]
        for i in range(batch_size):
            if ft[i].item():
                for p in [0, 1]:  # for ON and OFF separately
                    mu = np.random.uniform(th_mu_min_list[p], th_mu_max_list[p])
                    std = np.random.uniform(th_std_min, th_std_max)
                    self.threshold_mus[i, p] = mu
                    self.thresholds[p, i].normal_(mean=mu, std=std)
                    self.thresholds[p, i].clamp_(0.01, 1.0)

    def randomize_cutoff(self, first_times, cutoff_min=0, cutoff_max=900):
        """
        Randomizes the cutoff rates per video

        Args:
            first_times: B video just started flags
            cutoff_min: in hz
            cutoff_max: in hz
        """
        ft = first_times
        cutoff_rates = torch.zeros_like(self.cutoff_rates).uniform_(cutoff_min, cutoff_max)
        self.cutoff_rates = ft * cutoff_rates + (1 - ft) * self.cutoff_rates

    def randomize_leak(self, first_times, leak_min=0, leak_max=1):
        """
        Randomizes the leak rates per video

        Args:
            first_times: B video just started flags
            leak_min: in hz
            leak_max: in hz
        """
        ft = first_times
        rates = torch.zeros_like(self.leak_rates).uniform_(leak_min * 1e-6, leak_max * 1e-6)
        self.leak_rates = ft * rates + (1 - ft) * self.leak_rates

    def randomize_shot(self, first_times, shot_min=0, shot_max=1):
        """
        Randomizes the shot noise per video

        Args:
            shot_min: in hz
            shot_max: in hz
        """
        ft = first_times
        rates = torch.zeros_like(self.shot_rates).uniform_(shot_min * 1e-6, shot_max * 1e-6)
        self.shot_rates = ft * rates + (1 - ft) * self.shot_rates

    def randomize_refractory_periods(self, first_times, ref_min=10, ref_max=1000):
        """
        Randomizes the refractory period per video

        Args:
            first_times: B video just started flags
            ref_min: in microseconds
            ref_max: in microseconds
        """
        ft = first_times
        rates = torch.zeros_like(self.shot_rates).uniform_(ref_min, ref_max)
        self.refractory_periods = ft * rates + (1 - ft) * self.refractory_periods

    def forward(self):
        raise NotImplementedError

    def _kernel_call(self, log_images, video_len, image_ts, first_times, cuda_kernel, cpu_kernel, args_list, *args,
                     reset_rng_states=True):
        """
        generic functions to call simulation and feature computation kernels.

        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            cuda_kernel (function): numba.cuda jitted function (defined in events_kernel.py)
            cpu_kernel (function): numba jitted function (defined in events_kernel.py)
            args_list (Tensor list): additional Tensor arguments that the kernel might take as argument.
            *args: additional flags for the kernel
        """
        device = log_images.device
        height, width = log_images.shape[:2]
        batch_size = len(video_len)

        if reset_rng_states:
            self.rng_states.uniform_()
        # prepare args
        args_list = args_list + [
            log_images, video_len.cumsum(0), image_ts, first_times, self.rng_states, self.log_states,
            self.prev_log_images, self.timestamps, self.thresholds, self.prev_image_ts, self.refractory_periods,
            self.leak_rates, self.shot_rates, self.threshold_mus]

        if device.type == "cuda":
            args_list = [v.to(device) for v in args_list]
            cu_args = [cuda.as_cuda_array(v) for v in args_list] + list(args)

            block_dim = (1, 16, 16)
            sizes = (batch_size, height, width)
            grid_dim = tuple(int(np.ceil(a / b)) for a, b in zip(sizes, block_dim))

            # kernel
            cuda_kernel[grid_dim, block_dim](*cu_args)
        else:
            args_list = [v.numpy() for v in args_list] + list(args)

            # kernel
            cpu_kernel(*args_list)

    @torch.no_grad()
    def get_events(self, log_images, video_len, image_ts, first_times):
        """
        Retrieves the AER event list in a pytorch array.

        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
        Returns:
            events: N,5 in batch_index, x, y, polarity, timestamp (micro-seconds)
        """
        height, width = log_images.shape[:2]
        batch_size = len(video_len)

        self.counts[...] = 0

        self._kernel_call(log_images, video_len, image_ts, first_times, _cuda_kernel_count_events,
                          _cpu_kernel_count_events, [self.counts], False, reset_rng_states=True)
        # come-up with offset
        event_counts = self.counts
        cumsum = event_counts.view(-1).cumsum(dim=0)
        total_num_events = cumsum[-1].item()
        offsets = cumsum.view(batch_size, height, width) - event_counts

        events = torch.full((total_num_events, 5), fill_value=-10, device=cumsum.device, dtype=torch.int32)

        self._kernel_call(log_images, video_len, image_ts, first_times, _cuda_kernel_fill_events,
                          _cpu_kernel_fill_events, [events, offsets], True, reset_rng_states=False)

        # update values
        self.prev_image_ts = image_ts[:, -1]

        return events

    @torch.no_grad()
    def count_events(self, log_images, video_len, image_ts, first_times, reset=True, persistent=True):
        """
        Estimates the number of events per pixel.

        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            reset: do reset the count variable
        Returns:
            counts: B,H,W
        """

        if reset:
            self.counts[...] = 0

        self._kernel_call(log_images, video_len, image_ts, first_times, _cuda_kernel_count_events,
                          _cpu_kernel_count_events, [self.counts], persistent, reset_rng_states=True)

        # update values
        self.prev_image_ts = image_ts[:, -1]

        return self.counts

    @torch.no_grad()
    def event_volume(self, log_images, video_len, image_ts, first_times, nbins, mode='bilinear', split_channels=False):
        """
        Computes a volume of discretized images formed after the events, without
        storing the AER events themselves. We go from simulation directly to this
        space-time quantized representation. You can obtain the event-volume of
        [Unsupervised Event-based Learning of Optical Flow, Zhu et al. 2018] by
        specifying the mode to "bilinear" or you can obtain a stack of histograms
        if mode is set to "nearest".

        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            nbins (int): number of time-bins for the voxel grid
            mode (str): bilinear or nearest
            split_channels: if True positive and negative events have a distinct channels instead of doing their
                difference in a single channel.
        """
        prev_times = self.prev_image_ts * (1 - first_times) + image_ts[:, 0] * first_times
        end_times = image_ts[:, -1]

        target_timestamps = torch.cat((prev_times[:, None], end_times[:, None]), 1).long()

        return self.event_volume_sequence(log_images, video_len, image_ts, target_timestamps,
                                          first_times, nbins, mode, split_channels).squeeze(0)

    @torch.no_grad()
    def event_volume_sequence(
            self,
            log_images,
            video_len,
            image_ts,
            target_timestamps,
            first_times,
            nbins,
            mode='bilinear',
            split_channels=False):
        """
        Computes a volume of discretized images formed after the events, without
        storing the AER events themselves. We go from simulation directly to this
        space-time quantized representation. You can obtain the event-volume of
        [Unsupervised Event-based Learning of Optical Flow, Zhu et al. 2018] by
        specifying the mode to "bilinear" or you can obtain a stack of histograms
        if mode is set to "nearest".
        Here, we also receive a sequence of target timestamps to cut non uniformly the event volumes.

        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            nbins (int): number of time-bins for the voxel grid
            mode (str): bilinear or nearest
            split_channels: if True positive and negative events have a distinct channels instead of doing their
                difference in a single channel.
        """
        height, width = log_images.shape[:2]
        num_channels = nbins * 2 if split_channels else nbins
        batch_size = len(video_len)
        batch_times = target_timestamps.shape[1] - 1

        device = log_images.device
        voxel_grid = torch.zeros(
            (batch_times, batch_size, num_channels, height, width),
            dtype=torch.float32, device=device)

        # arbitrary voxel start times and durations
        voxel_start_times = self.prev_image_ts * (1 - first_times) + image_ts[:, 0] * first_times
        voxel_durations = image_ts[:, -1] - voxel_start_times

        # prepare args
        args = [voxel_grid, target_timestamps]

        self._kernel_call(
            log_images, video_len, image_ts, first_times, _cuda_kernel_voxel_grid_sequence,
            _cpu_kernel_voxel_grid_sequence, args, True, mode == 'bilinear', split_channels, reset_rng_states=True)

        self.prev_image_ts = image_ts[:, -1]

        voxel_grid = voxel_grid.view(batch_times, batch_size, num_channels, height, width)
        return voxel_grid


    @torch.no_grad()
    def event_volume_and_ts_sequence(
            self,
            log_images,
            video_len,
            image_ts,
            target_timestamps,
            first_times,
            nbins,
            mode='bilinear',
            split_channels=False):
        """

        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            nbins (int): number of time-bins for the voxel grid
            mode (str): bilinear or nearest
            split_channels: if True positive and negative events have a distinct channels instead of doing their
                difference in a single channel.
        """
        height, width = log_images.shape[:2]
        num_channels = nbins * 2 if split_channels else nbins
        batch_size = len(video_len)
        batch_times = target_timestamps.shape[1] - 1

        device = log_images.device
        voxel_grid = torch.zeros(
            (batch_times, batch_size, num_channels, height, width),
            dtype=torch.float32, device=device)
        time_surface = torch.zeros(
            (batch_times, batch_size, height, width),
            dtype=torch.float32, device=device)
        
        # arbitrary voxel start times and durations
        voxel_start_times = self.prev_image_ts * (1 - first_times) + image_ts[:, 0] * first_times
        voxel_durations = image_ts[:, -1] - voxel_start_times

        # prepare args
        args = [voxel_grid, time_surface, target_timestamps]

        self._kernel_call(
            log_images, video_len, image_ts, first_times, _cuda_kernel_voxel_grid_and_ts_sequence,
            _cpu_kernel_voxel_grid_and_ts_sequence, args, True, mode == 'bilinear', split_channels, reset_rng_states=True)

        self.prev_image_ts = image_ts[:, -1]

        voxel_grid = voxel_grid.view(batch_times, batch_size, num_channels, height, width)
        time_surface = time_surface.view(batch_times, batch_size, height, width) + 127.5
        return voxel_grid, time_surface

    ####time surface
    @torch.no_grad()
    def event_time_surface_sequence(
            self,
            log_images,
            video_len,
            image_ts,
            target_timestamps,
            first_times,
            nbins,
            mode='bilinear',
            split_channels=False):
        """
        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            nbins (int): number of time-bins for the voxel grid
            mode (str): bilinear or nearest
            split_channels: if True positive and negative events have a distinct channels instead of doing their
                difference in a single channel.
        """
        height, width = log_images.shape[:2]
        num_channels = nbins * 2 if split_channels else nbins
        batch_size = len(video_len)
        batch_times = target_timestamps.shape[1] - 1

        device = log_images.device
        voxel_grid = torch.zeros(
            (batch_times, batch_size, num_channels, height, width),
            dtype=torch.float32, device=device)

        # arbitrary voxel start times and durations
        voxel_start_times = self.prev_image_ts * (1 - first_times) + image_ts[:, 0] * first_times
        voxel_durations = image_ts[:, -1] - voxel_start_times

        #create count mask
        ######1
        # q1 = event_counts.quantile(0.25)
        # q3 = event_counts.quantile(0.75)
        # iqr = q3 - q1

        # lower = q1 - 1.5 * iqr
        # upper = q3 + 1.5 * iqr

        # filtered = event_counts[(event_counts >= lower) & (event_counts <= upper)]


        ######2
        # p10 = event_counts.quantile(0.05)
        # p90 = event_counts.quantile(0.95)

        # 只保留 P10 ~ P90 之间的数据
        # mask = (event_counts >= p10) & (event_counts <= p90)
        # filtered = event_counts[mask]

        # robust_mean = filtered.mean()
        # robust_std = filtered.std(unbiased=False)

        # threshold = robust_mean - 1*robust_std
        # mask = (event_counts > threshold).int()
        # mask = mask.unsqueeze(0).unsqueeze(2).expand(batch_times, batch_size, num_channels, height, width)


        # prepare args
        args = [voxel_grid, target_timestamps, self.counts]

        self._kernel_call(
            log_images, video_len, image_ts, first_times, _cuda_kernel_time_surface_sequence,
            _cpu_kernel_time_surface_sequence, args, False, reset_rng_states=True)
        
        event_counts = self.counts

        threshold = 5
        mask = (event_counts > threshold).int()
        mask = mask.unsqueeze(0).unsqueeze(2)

        self.prev_image_ts = image_ts[:, -1]

        voxel_grid = voxel_grid.view(batch_times, batch_size, num_channels, height, width)
        voxel_grid = voxel_grid * mask + 128
        return voxel_grid


####time surface
    @torch.no_grad()
    def event_time_surface3D_sequence(
            self,
            log_images,
            video_len,
            image_ts,
            target_timestamps,
            first_times,
            nbins,
            mode='bilinear',
            split_channels=False):
        """
        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            nbins (int): number of time-bins for the voxel grid
            mode (str): bilinear or nearest
            split_channels: if True positive and negative events have a distinct channels instead of doing their
                difference in a single channel.
        """
        height, width = log_images.shape[:2]
        num_channels = nbins * 3 
        batch_size = len(video_len)
        batch_times = target_timestamps.shape[1] - 1

        device = log_images.device
        voxel_grid = torch.zeros(
            (batch_times, batch_size, num_channels, height, width),
            dtype=torch.float32, device=device)

        # arbitrary voxel start times and durations
        voxel_start_times = self.prev_image_ts * (1 - first_times) + image_ts[:, 0] * first_times
        voxel_durations = image_ts[:, -1] - voxel_start_times
       
        # prepare args
        args = [voxel_grid, target_timestamps, self.counts]

        self._kernel_call(
            log_images, video_len, image_ts, first_times, _cuda_kernel_time_surface3D_sequence,
            _cpu_kernel_time_surface3D_sequence, args, False, reset_rng_states=True)
        
        event_counts = self.counts

        threshold = 5
        mask = (event_counts > threshold).int()
        mask = mask.unsqueeze(0).unsqueeze(2)

        self.prev_image_ts = image_ts[:, -1]

        voxel_grid = voxel_grid.view(batch_times, batch_size, num_channels, height, width)
        voxel_grid = voxel_grid * mask
        return voxel_grid


####DA表示方法
    @torch.no_grad()
    def event_volume_DA_sequence(
            self,
            log_images,
            video_len,
            image_ts,
            target_timestamps,
            first_times,
            nbins,
            mode='nearest',
            split_channels=False):
        """
        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            nbins (int): number of time-bins for the voxel grid
            mode (str): bilinear or nearest
            split_channels: if True positive and negative events have a distinct channels instead of doing their
                difference in a single channel.
        """
        height, width = log_images.shape[:2]
        num_channels = nbins * 2 if split_channels else nbins
        batch_size = len(video_len)
        batch_times = target_timestamps.shape[1] - 1

        device = log_images.device
        voxel_grid = torch.zeros(
            (batch_times, batch_size, num_channels, height, width),
            dtype=torch.float32, device=device)
        DA_grid = torch.zeros(
            (batch_times, batch_size, num_channels, height, width),
            dtype=torch.float32, device=device)
        num_grid = torch.zeros(
            (batch_times, batch_size, num_channels, height, width),
            dtype=torch.float32, device=device)

        # arbitrary voxel start times and durations
        voxel_start_times = self.prev_image_ts * (1 - first_times) + image_ts[:, 0] * first_times
        voxel_durations = image_ts[:, -1] - voxel_start_times

        # prepare args
        args = [voxel_grid, DA_grid, num_grid, target_timestamps]

        self._kernel_call(
            log_images, video_len, image_ts, first_times, _cuda_kernel_voxel_grid_DA_sequence,
            _cpu_kernel_voxel_grid_DA_sequence, args, True, mode == 'bilinear', split_channels, reset_rng_states=True)

        self.prev_image_ts = image_ts[:, -1]

        voxel_grid = voxel_grid.view(batch_times, batch_size, num_channels, height, width)
        num_grid = num_grid.view(batch_times, batch_size, num_channels, height, width)
        DA_grid = DA_grid.view(batch_times, batch_size, num_channels, height, width)

        mask = num_grid > 0
        voxel_grid[mask] = voxel_grid[mask]/ num_grid[mask]
        return voxel_grid

####TenCode
    @torch.no_grad()
    def event_volume_TenCode_sequence(
            self,
            log_images,
            video_len,
            image_ts,
            target_timestamps,
            first_times,
            nbins,
            mode='nearest',
            split_channels=False):
        """
        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            nbins (int): number of time-bins for the voxel grid
            mode (str): bilinear or nearest
            split_channels: if True positive and negative events have a distinct channels instead of doing their
                difference in a single channel.
        """
        height, width = log_images.shape[:2]
        num_channels = nbins * 3 #RGB
        batch_size = len(video_len)
        batch_times = target_timestamps.shape[1] - 1

        device = log_images.device
        voxel_grid = torch.zeros(
            (batch_times, batch_size, num_channels, height, width),
            dtype=torch.float32, device=device)
        num_grid = torch.zeros(
            (batch_times, batch_size, num_channels, height, width),
            dtype=torch.float32, device=device)

        # arbitrary voxel start times and durations
        voxel_start_times = self.prev_image_ts * (1 - first_times) + image_ts[:, 0] * first_times
        voxel_durations = image_ts[:, -1] - voxel_start_times

        # prepare args
        args = [voxel_grid, num_grid, target_timestamps]

        self._kernel_call(
            log_images, video_len, image_ts, first_times, _cuda_kernel_voxel_grid_TenCode_sequence,
            _cpu_kernel_voxel_grid_TenCode_sequence, args, True, mode == 'bilinear', split_channels, reset_rng_states=True)

        self.prev_image_ts = image_ts[:, -1]

        voxel_grid = voxel_grid.view(batch_times, batch_size, num_channels, height, width)

        ###count filter
        threshold = 2
        mask = (num_grid >= threshold).int()
        voxel_grid = voxel_grid * mask

        ##reshape (rrr..., ggg..., bbb...,) to (rgb,rgb,rgb, ...)
        voxel_grid = voxel_grid.view(batch_times, batch_size, nbins, 3,  height, width).permute(0,1,3,2,4,5).reshape(batch_times, batch_size, 3*nbins, height, width)
        return voxel_grid


    @torch.no_grad()
    def event_volume_sequence_with_time_and_number(
            self,
            log_images,
            video_len,
            image_ts,
            target_timestamps,
            first_times,
            nbins,
            mode='bilinear',
            split_channels=False):
        """
        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            nbins (int): number of time-bins for the voxel grid
            mode (str): bilinear or nearest
            split_channels: if True positive and negative events have a distinct channels instead of doing their
                difference in a single channel.
        """
        height, width = log_images.shape[:2]
        num_channels = nbins * 2 if split_channels else nbins
        batch_size = len(video_len)
        batch_times = target_timestamps.shape[1] - 1

        ##增加的events部分
        self.counts[...] = 0

        self._kernel_call(log_images, video_len, image_ts, first_times, _cuda_kernel_count_events,
                          _cpu_kernel_count_events, [self.counts], False, reset_rng_states=True)
        # come-up with offset
        event_counts = self.counts
        cumsum = event_counts.view(-1).cumsum(dim=0)
        total_num_events = cumsum[-1].item()
        offsets = cumsum.view(batch_size, height, width) - event_counts

        events_with_tc = torch.full((total_num_events, 7), fill_value=-10, device=cumsum.device, dtype=torch.int32) # events with t,b,c,x,y,p,t

        device = log_images.device
        voxel_grid = torch.zeros(
            (batch_times, batch_size, num_channels, height, width,5),
            dtype=torch.float32, device=device) # voxel with x_sum,y_sum,p_sum,t_sum,num

        # arbitrary voxel start times and durations
        voxel_start_times = self.prev_image_ts * (1 - first_times) + image_ts[:, 0] * first_times
        voxel_durations = image_ts[:, -1] - voxel_start_times

        # prepare args
        args = [voxel_grid, target_timestamps, events_with_tc, offsets]

        self._kernel_call(
            log_images, video_len, image_ts, first_times, _cuda_kernel_voxel_grid_sequence_5,
            _cpu_kernel_voxel_grid_sequence_5, args, True, mode == 'bilinear', split_channels, reset_rng_states=True)

        self.prev_image_ts = image_ts[:, -1]

        voxel_grid = voxel_grid.view(batch_times, batch_size, num_channels, height, width,5)
        return voxel_grid, events_with_tc

    @torch.no_grad()
    def get_events_with_bin(self, log_images, video_len, image_ts, first_times):
        """
        Retrieves the AER event list in a pytorch array.

        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
        Returns:
            events: N,T,C,5 in batch_index, x, y, polarity, timestamp (micro-seconds)
        """
        height, width = log_images.shape[:2]
        batch_size = len(video_len)

        self.counts[...] = 0

        self._kernel_call(log_images, video_len, image_ts, first_times, _cuda_kernel_count_events,
                          _cpu_kernel_count_events, [self.counts], False, reset_rng_states=True)
        # come-up with offset
        event_counts = self.counts
        cumsum = event_counts.view(-1).cumsum(dim=0)
        total_num_events = cumsum[-1].item()
        offsets = cumsum.view(batch_size, height, width) - event_counts

        events = torch.full((total_num_events, 5), fill_value=-10, device=cumsum.device, dtype=torch.int32)

        self._kernel_call(log_images, video_len, image_ts, first_times, _cuda_kernel_fill_events,
                          _cpu_kernel_fill_events, [events, offsets], True, reset_rng_states=False)

        # update values
        self.prev_image_ts = image_ts[:, -1]

        return events


    @torch.no_grad()
    def log_images(self, u8imgs, eps=1e-7):
        """
        Converts byte images to log

        Args:
            u8imgs (torch.Tensor): B,C,H,W,T byte images
            eps (float): epsilon factor
        """
        return torch.log(u8imgs.float() / 255.0 + eps)

    @torch.no_grad()
    def dynamic_moving_average(self, images, num_frames, timestamps, first_times, min_pixel_range=20,
                               max_pixel_incr=20, eps=1e-7):
        """

        Converts byte images to log and
        performs a pass-band motion blur of incoming images.
        This simulates the latency of the photodiode w.r.t to incoming
        light dynamic.

        Args:
            images (torch.Tensor): H,W,T byte or float images in the 0 to 255 range
            num_frames (torch.Tensor): shape (B,) len of each video in the batch.
            timestamps (torch.Tensor): B,T timestamps
            first_times (torch.Tensor): B flags
            eps (float): epsilon factor
        """
        fl_images = images.float()

        if self.cutoff_rates.sum() == 0:  # this broke the sample
            return self.log_images(images, eps=eps)

        height, width = images.shape[:2]
        batch_size = len(num_frames)

        args = [fl_images, num_frames.cumsum(0), self.prev_log_images_0, self.prev_log_images_1,
                self.filtering_prev_image_ts, timestamps, first_times, self.cutoff_rates]

        if images.device.type == "cuda":
            cu_args = [cuda.as_cuda_array(v) for v in args]
            block_dim = (1, 16, 16)
            sizes = (batch_size, height, width)
            grid_dim = tuple(int(np.ceil(a / b)) for a, b in zip(sizes, block_dim))
            _cuda_kernel_dynamic_moving_average[grid_dim, block_dim](*cu_args, min_pixel_range, max_pixel_incr, eps)
        else:
            args = [v.numpy() for v in args]
            _cpu_kernel_dynamic_moving_average(*args, min_pixel_range, max_pixel_incr, eps)

        self.filtering_prev_image_ts = timestamps[:, -1]

        return fl_images
