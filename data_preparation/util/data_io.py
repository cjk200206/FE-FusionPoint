import cv2
from enum import Enum
import glob
import h5py
import numpy as np
import os
from pandas import read_csv
from scipy.sparse import csr_matrix

RequiredData = Enum("DataSplit", ["events", "event_time_range", "images", "image_stamps", "calib"])

# ----- High level dataset loading -----
def load_dataset(dataset, path, required_data):
    assert required_data

    # Event time range will be only computed if not all events are required
    if RequiredData.events in required_data:
        only_event_time_range = False
        if RequiredData.event_time_range in required_data:
            required_data.remove(RequiredData.event_time_range)
    if RequiredData.event_time_range in required_data:
        only_event_time_range = True
        required_data.append(RequiredData.events)

    supported_datasets = ["ecd", "fpv", "mvsec", "vivid", "ddd20", "griffin", "hdr", "sde","gopro", "reblur","uevd","gen_coco", "gen_coco_1"]
    events = images = image_stamps = calib = 0
    if dataset == supported_datasets[0] or \
       dataset == supported_datasets[3] or \
       dataset == supported_datasets[5]:  # ecd, converted rosbags (vivid, griffin)
        if RequiredData.events in required_data:
            events = load_events_txt(os.path.join(path, "events.txt"), only_event_time_range)
        if RequiredData.images in required_data: 
            images = load_images_ecd(os.path.join(path, "images"))
        if RequiredData.image_stamps in required_data:
            image_stamps = load_image_timestamps_ecd(os.path.join(path, "images.txt"))
        if RequiredData.calib in required_data:
            calib = load_calib_txt(os.path.join(path, "calib.txt"))
    elif dataset == supported_datasets[1]:  # fpv
        if RequiredData.events in required_data:
            events = load_events_txt(os.path.join(path, "events.txt"), only_event_time_range)
        if RequiredData.images in required_data or RequiredData.image_stamps in required_data:
            images, image_stamps = load_images_and_stamps_fpv(os.path.join(path, "images.txt"))
        if RequiredData.calib in required_data:
            calib = load_calib_txt(os.path.join(path, "calib.txt"))
    elif dataset == supported_datasets[2]:  # mvsec
        if RequiredData.events in required_data or RequiredData.images in required_data or RequiredData.image_stamps in required_data:
            hdf5_file_list = glob.glob(os.path.join(path, "*.hdf5"))
            assert len(hdf5_file_list) == 1, "There should only be one .hdf5 file in folder " + path
            events, images, image_stamps = load_data_mvsec(hdf5_file_list[0], only_event_time_range)
        if RequiredData.calib in required_data:
            calib = load_calib_txt(os.path.join(path, "calib.txt"))
    elif dataset == supported_datasets[4]:  # ddd20
        if RequiredData.events in required_data or RequiredData.images in required_data or RequiredData.image_stamps in required_data:
            hdf5_file_list = glob.glob(os.path.join(path, "*.hdf5.exported.hdf5"))  # data must be exported first
            assert len(hdf5_file_list) == 1, "There should only be one .exported.hdf5 file in folder " + path
            events, images, image_stamps = load_data_ddd20(hdf5_file_list[0], only_event_time_range)
        if RequiredData.calib in required_data:
            calib = load_calib_txt(os.path.join(path, "calib.txt"))
    elif dataset == supported_datasets[6]:  # hdr
        mat_file_list = glob.glob(os.path.join(path, "*.mat"))
        assert len(mat_file_list) == 1, "There should only be one .mat file in folder " + path
        events, images, image_stamps, calib = load_data_hdr(mat_file_list[0], only_event_time_range)
    elif dataset == supported_datasets[7]:  # sde
        events, images, image_stamps, calib = load_data_sde(path, only_event_time_range)
    elif dataset == supported_datasets[8]:  # gopro
        hdf5_file_list = glob.glob(os.path.join(path, "*.h5"))
        assert len(hdf5_file_list) == 1, "There should only be one .h5 file in folder " + path
        events, images, image_stamps, calib = load_data_gopro(hdf5_file_list[0], only_event_time_range)
    elif dataset == supported_datasets[9]:  # reblur
        hdf5_file_list = glob.glob(os.path.join(path, "*.h5"))
        assert len(hdf5_file_list) == 1, "There should only be one .h5 file in folder " + path
        events, images, image_stamps, calib = load_data_reblur(hdf5_file_list[0], only_event_time_range)
    elif dataset == supported_datasets[10]:  # uevd
        # events, images, image_stamps, calib = load_data_uevd(path, only_event_time_range)
        events, images, image_stamps, calib = load_data_uevd_custom_blur(path, only_event_time_range)
    elif dataset == supported_datasets[11] or \
        dataset == supported_datasets[12]:  # gen_coco
        events, images, image_stamps, calib = load_data_gen_coco_blur(path, only_event_time_range)
    else:
        raise NotImplementedError("Dataset", dataset, " is not supported. Supported datasets:", supported_datasets)
    
    # Shift time to start with first event
    if RequiredData.events in required_data and RequiredData.event_time_range not in required_data and RequiredData.image_stamps in required_data:
        start_time = events[0, 0]
        events[:, 0] = events[:, 0] - start_time
        image_stamps = image_stamps - start_time
    
    # Sanity checks
    if RequiredData.event_time_range in required_data:
        assert events.shape == (2,)
        assert events[0] < events[1], str(events)
    elif RequiredData.events in required_data:  # this is the setting to load all events (without RequiredData.event_time_range)
        assert events.shape[1] == 4
    if RequiredData.images in required_data:
        assert len(images) > 0
    if RequiredData.image_stamps in required_data:
        assert len(image_stamps.shape) == 1
    if RequiredData.calib in required_data:
        assert len(calib) == 9 or len(calib) == 8

    return events, images, image_stamps, calib

def load_dataset_images_calib(dataset, path):
    supported_datasets = ["ecd", "fpv"]
    if dataset == supported_datasets[0]:  # ecd
        images = load_images_ecd(os.path.join(path, "images"))
        calib = load_calib_txt(os.path.join(path, "calib.txt"))
    elif dataset == supported_datasets[1]:  # fpv
        images, _ = load_images_and_stamps_fpv(os.path.join(path, "images.txt"))
        calib = load_calib_txt(os.path.join(path, "calib.txt"))
    else:
        raise NotImplementedError("Dataset", dataset, " is not supported. Supported datasets:", supported_datasets)
    
    # Sanity checks
    assert len(calib) == 9 or len(calib) == 8

    return images, calib


# ----- General .txt files -----
def load_calib_txt(path):
    calib_data = np.genfromtxt(path)
    print("Calibration loaded")
    return calib_data

def load_events_txt(path, only_event_time_range=False):
    print("Loading events from", path)

    if only_event_time_range:
        num_lines = sum(1 for _ in open(path))
        events = read_csv(path, header='infer', delimiter=" ", usecols=[0], skiprows=range(2, num_lines - 1)).to_numpy()
        events = events.flatten()
        print("Events in time range", events)
    else:  # load all events
        events = read_csv(path, header='infer', delimiter=" ", usecols=range(4)).to_numpy()
    
    print("Raw events loaded")
    return events


# ----- Event camera dataset ------
def load_images_ecd(path):
    print("Loading images from" , path, "...")
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in sorted(glob.glob(os.path.join(path, '*.png')))]
    print("Loaded", len(images), "images.")
    return images

def load_image_timestamps_ecd(path):
    timestamps = np.genfromtxt(path, usecols=[0])
    print("Loaded image timestamps")
    return timestamps


# ------ RPG FPV drone racing dataset -----
def load_images_and_stamps_fpv(path):
    # Load data from images.txt
    print("Loading data from", path)
    data = read_csv(path, header='infer', delimiter=" ")
    data_keys = data.keys()
    image_timestamps = data[data_keys[1]].to_numpy()
    image_path_list = data[data_keys[2]]
    
    # Load images
    images = [cv2.imread(os.path.join(os.path.dirname(path), img_path), cv2.IMREAD_GRAYSCALE) for img_path in image_path_list]
    print("Loaded", len(images), "images.")
    return images, image_timestamps

# ------ MVSEC h5-file -----
def load_data_mvsec(path, only_event_time_range=False):
    assert os.path.isfile(path)

    data = h5py.File(path, "r")["davis"]["left"]

    if only_event_time_range:
        events = np.array([data["events"][0][2], data["events"][-1][2]])
    else:  # load all events
        events = np.array(data["events"])
        events = events[:, [2, 0, 1, 3]]  # fix order: t, x, y, p
        events[:, -1] = np.clip(events[:, -1], 0, 1)

    images = [np.array(img) for img in data["image_raw"]]
    image_timestamps = np.array(data["image_raw_ts"])

    return events, images, image_timestamps

# ------ DDD20 exported h5-file -----
def load_data_ddd20(path, only_event_time_range=False):
    assert os.path.isfile(path)

    data = h5py.File(path, "r")

    # Detect if data is partially corrupted
    idx_max_event_timestamp = np.argmax(data["event"][:, 0])
    if idx_max_event_timestamp + 1 < len(data["event"]):
        print("Warning: Sequence", path, "seems to have partially corrupted timestamps. Only the first", (idx_max_event_timestamp + 1) / len(data["event"]) * 100, "% of the data can be used.")
        if only_event_time_range:
            events = np.array([data["event"][0][0], data["event"][idx_max_event_timestamp][0]]) * 1e-6
        else:
            events = np.array(data["event"][:idx_max_event_timestamp + 1], dtype=np.float32)
            events[:, 0] = events[:, 0] * 1e-6  # convert event timestamp to s

        idx_max_frame_timestamp = np.argmax(data["frame_ts"])
        images = [np.array(img) for img in data["frame"][:idx_max_frame_timestamp + 1]]
        image_timestamps = np.array(data["frame_ts"][:idx_max_frame_timestamp + 1]).reshape([-1])

    else:
        # Timestamps not corrupted
        if only_event_time_range:
            events = np.array([data["event"][0][0], data["event"][-1][0]]) * 1e-6 
        else:  # load all events
            events = np.array(data["event"], dtype=np.float32)
            events[:, 0] = events[:, 0] * 1e-6  # convert event timestamp to s

        images = [np.array(img) for img in data["frame"]]
        image_timestamps = np.array(data["frame_ts"]).reshape([-1])

    return events, images, image_timestamps

# ------ HDR exported h5-file -----
def load_data_hdr(path, only_event_time_range=False):
    assert os.path.isfile(path)

    data = h5py.File(path, "r")

    if only_event_time_range:
        events = np.array([data["events"][0][0], data["event"][0][-1]], dtype=np.float32) * 1e-6 
    else:  # load all events
        events = np.array(data["events"], dtype=np.float32).transpose()
        events[:, 0] = events[:, 0] * 1e-6  # convert event timestamp to s
        # Fix matlab indexing
        events[:, 1] = events[:, 1] - 1
        events[:, 2] = events[:, 2] - 1

    images = [np.array(img).transpose() for img in data["image"]]
    image_timestamps = np.array(data["time_image"]).reshape([-1]) * 1e-6

    calib = np.array([1., 1., 0., 0., 0., 0., 0., 0., 0.])  # Skip undistortion

    return events, images, image_timestamps, calib

# ------ SDE dataset ----- #
def load_data_sde(path, only_event_time_range=False):
    print("Loading images from" , path, "...")
    normal_images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in sorted(glob.glob(os.path.join(path, 'normal', '*.png')))]
    dark_images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in sorted(glob.glob(os.path.join(path, 'low', '*.png')))]

    if only_event_time_range:
        npz_loaded = np.load(os.path.join(path, 'low', 'lowlight_event.npz'))
        events = np.array([npz_loaded['arr_0'][0][0], npz_loaded['arr_0'][-1][0]])*1e-6
    else:  # load all events
        npz_loaded = np.load(os.path.join(path, 'low', 'lowlight_event.npz'))
        events_raw = np.array(npz_loaded['arr_0'][:])
        ts, x, y, p = [events_raw[key] for key in ['timestamp', 'x', 'y', 'polarity']]
        ts = ts * 1e-6  # convert to seconds
        events = np.stack([ts, x, y, p], axis=1)

    image_timestamps = np.array([int(os.path.splitext(os.path.basename(f_name))[0]) 
                                 for f_name in sorted(glob.glob(os.path.join(path, 'low', '*.png')))]) * 1e-6
    
    calib = np.array([1., 1., 0., 0., 0., 0., 0., 0., 0.])  # Skip undistortion

    return events, (normal_images, dark_images), image_timestamps, calib

# ------ Gopro h5-file -----
def load_data_gopro(path, only_event_time_range=False):
    assert os.path.isfile(path)

    data = h5py.File(path, "r")

    if only_event_time_range:
        t = data["events"]["ts"][:] * 1e-9  # convert to seconds
        events = np.array([t[0], t[-1]]) 

    else:  # load all events
        x = data["events"]["xs"][:]
        y = data["events"]["ys"][:]
        t = data["events"]["ts"][:] * 1e-9  # convert to seconds
        p = data["events"]["ps"][:]
        events = np.stack([t, x, y, p], axis=1)

        events[:, -1] = np.clip(events[:, -1], 0, 1)

    sharp_images = []
    images = []
    image_timestamps = []
    for img_name in data['images']:
            images.append(data['images/{}'.format(img_name)][:])
            sharp_images.append(data['sharp_images/{}'.format(img_name)][:])
            image_timestamps.append(data['images/{}'.format(img_name)].attrs['timestamp'])
    sharp_images = np.array(sharp_images)
    images = np.array(images)
    image_timestamps = np.array(image_timestamps) * 1e-9  # convert to seconds

    calib = np.array([1., 1., 0., 0., 0., 0., 0., 0., 0.])  # Skip undistortion

    return events, (images, sharp_images), image_timestamps, calib

# ------ REBlur h5-file -----
def load_data_reblur(path, only_event_time_range=False):
    assert os.path.isfile(path)

    data = h5py.File(path, "r")

    if only_event_time_range:
        t = data["events"]["ts"][:] * 1e-9  # convert to seconds
        events = np.array([t[0], t[-1]]) 

    else:  # load all events
        x = data["events"]["xs"][:]
        y = data["events"]["ys"][:]
        t = data["events"]["ts"][:] * 1e-9  # convert to seconds
        p = data["events"]["ps"][:]
        events = np.stack([t, x, y, p], axis=1)

        events[:, -1] = np.clip(events[:, -1], 0, 1)

    sharp_images = []
    images = []
    image_timestamps = []
    for img_name in data['images']:
            images.append(data['images/{}'.format(img_name)][:])
            sharp_images.append(data['sharp_images/{}'.format(img_name)][:])
            image_timestamps.append(data['images/{}'.format(img_name)].attrs['timestamp'])
    sharp_images = np.array(sharp_images)
    images = np.array(images)
    image_timestamps = np.array(image_timestamps) * 1e-9  # convert to seconds

    calib = np.array([1., 1., 0., 0., 0., 0., 0., 0., 0.])  # Skip undistortion

    return events, (images, sharp_images), image_timestamps, calib

# ------ UEVD dataset ----- #
def load_data_uevd(path, only_event_time_range=False):
    print("Loading images from" , path, "...")
    if "train" in path:
        images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in sorted(glob.glob(os.path.join(path, 'blur_images','10','5-5','*.png')))]
        sharp_images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in sorted(glob.glob(os.path.join(path, 'clean_gt_images','10','5-5','*.png')))]
    else:
        images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in sorted(glob.glob(os.path.join(path, 'blur_images','14','9-5','*.png')))]
        sharp_images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in sorted(glob.glob(os.path.join(path, 'clean_gt_images','14','9-5','*.png')))]
    
    if only_event_time_range:
        npz_loaded = np.load(os.path.join(path, 'events', 'events.npz'))
        events = np.array([npz_loaded['data'][0][0], npz_loaded['data'][-1][0]])*1e-6
    else:  # load all events
        npz_loaded = np.load(os.path.join(path, 'events', 'events.npz'))
        events_raw = np.array(npz_loaded['data'][:])
        ts, x, y, p = [events_raw[key] for key in ['timestamp', 'x', 'y', 'polarity']]
        ts = ts * 1e-6  # convert to seconds
        events = np.stack([ts, x, y, p], axis=1)

    image_timestamps = np.loadtxt(os.path.join(path, 'frame_time.txt'),usecols=1) * 1e-6
    if "train" in path:
        image_timestamps = image_timestamps[2::10]
    else:
        image_timestamps = image_timestamps[4::14]

    image_timestamps = image_timestamps[:len(images)]    
    calib = np.array([1., 1., 0., 0., 0., 0., 0., 0., 0.])  # Skip undistortion

    return events, (images, sharp_images), image_timestamps, calib


def load_data_uevd_custom_blur(path, only_event_time_range=False):
    print("Loading raw frames for custom blur synthesis from", path, "...")
    
    # 1. 获取所有原始帧的路径
    all_frame_paths = sorted(glob.glob(os.path.join(path, 'frames', '*.png')))
    
    # 2. 定义 GT 的选取逻辑
    if "train" in path:
        ref_subdir = os.path.join('10', '5-5')
        step = 10
        start_idx = 2
        weights_3 = [0.2, 0.3, 0.5] 
    else:
        ref_subdir = os.path.join('14', '9-5')
        step = 14
        start_idx = 4
        weights_3 = [0.2, 0.3, 0.5]

    ref_gt_pattern = os.path.join(path, 'clean_gt_images', ref_subdir, '*.png')
    target_count = len(glob.glob(ref_gt_pattern))
    print(f"Target aligned quantity from clean_gt_images: {target_count}")

    num_frames = len(all_frame_paths)
    gt_indices = list(range(start_idx, num_frames, step))
    gt_indices = gt_indices[:target_count]
    
    # === 修改 1: 初始化多个模糊程度的列表 ===
    blur_images_3 = []
    blur_images_5 = []
    blur_images_7 = []
    blur_images_9 = []
    sharp_images = []

    print(f"Synthesizing {len(gt_indices)} blur pairs (3, 5, 7, 9 frames)...")

    # 辅助读取函数，避免重复代码
    def read_frames(indices):
        return [cv2.imread(all_frame_paths[i], cv2.IMREAD_GRAYSCALE) for i in indices]

    for gt_idx in gt_indices:
        if gt_idx + 1 >= num_frames:
            continue

        # --- 基础 3帧模糊 (原有逻辑) ---
        if gt_idx < 2: 
            print(f"Skipping frame {gt_idx}, not enough history.")
            continue
            
        # 读取 idx-2, idx-1, idx
        indices_3 = [gt_idx - 2, gt_idx - 1, gt_idx]
        frames_3 = read_frames(indices_3)
        
        if any(img is None for img in frames_3):
            continue

        gt_img = frames_3[-1] # 当前帧即为 GT
        
        # 合成 3帧模糊
        blur_3 = synthesize_blur_weighted(frames_3, weights=weights_3)
        
        # === 修改 2: 新增 5, 7, 9 帧模糊逻辑 ===
        # 策略：如果历史帧不够，直接复用 blur_3
        
        # --- 5帧模糊 ---
        if gt_idx >= 4:
            # 读取 idx-4, idx-3 ...
            # 优化：复用已读取的 frames_3，只读额外的
            extra_indices = [gt_idx - 4, gt_idx - 3]
            extra_frames = read_frames(extra_indices)
            if any(img is None for img in extra_frames):
                blur_5 = blur_3 # 读取失败降级
            else:
                frames_5 = extra_frames + frames_3
                # 权重：线性递增 [1, 2, 3, 4, 5]
                blur_5 = synthesize_blur_weighted(frames_5, weights=list(range(1, 6)))
        else:
            blur_5 = blur_3 # 帧数不够，用3帧替代

        # --- 7帧模糊 ---
        if gt_idx >= 6:
            extra_indices = [gt_idx - 6, gt_idx - 5]
            extra_frames_7 = read_frames(extra_indices)
            if any(img is None for img in extra_frames_7):
                blur_7 = blur_3
            else:
                # 注意：这里需要 5帧所有的图 + 新增的2帧。
                # 为了代码简单，如果 blur_5 是降级的，这里也会受影响，所以重新构建完整列表最稳妥
                # 完整 indices: idx-6 ... idx
                full_indices_7 = list(range(gt_idx - 6, gt_idx + 1))
                frames_7 = read_frames(full_indices_7)
                blur_7 = synthesize_blur_weighted(frames_7, weights=list(range(1, 8)))
        else:
            blur_7 = blur_3

        # --- 9帧模糊 ---
        if gt_idx >= 8:
            full_indices_9 = list(range(gt_idx - 8, gt_idx + 1))
            frames_9 = read_frames(full_indices_9)
            if any(img is None for img in frames_9):
                blur_9 = blur_3
            else:
                blur_9 = synthesize_blur_weighted(frames_9, weights=list(range(1, 10)))
        else:
            blur_9 = blur_3

        # 存入列表
        blur_images_3.append(blur_3)
        blur_images_5.append(blur_5)
        blur_images_7.append(blur_7)
        blur_images_9.append(blur_9)
        sharp_images.append(gt_img)

    # 3. 处理 Events (逻辑保持不变)
    if only_event_time_range:
        npz_loaded = np.load(os.path.join(path, 'events', 'events.npz'))
        events = np.array([npz_loaded['data'][0][0], npz_loaded['data'][-1][0]])*1e-6
    else:
        npz_loaded = np.load(os.path.join(path, 'events', 'events.npz'))
        events_raw = np.array(npz_loaded['data'][:])
        ts, x, y, p = [events_raw[key] for key in ['timestamp', 'x', 'y', 'polarity']]
        ts = ts * 1e-6  
        events = np.stack([ts, x, y, p], axis=1)

    # 4. 处理 Timestamps
    image_timestamps = np.loadtxt(os.path.join(path, 'frame_time.txt'), usecols=1) * 1e-6
    if "train" in path:
        image_timestamps = image_timestamps[start_idx::step]
    else:
        image_timestamps = image_timestamps[start_idx::step]
    image_timestamps = image_timestamps[:len(sharp_images)]    
    
    calib = np.array([1., 1., 0., 0., 0., 0., 0., 0., 0.]) 

    # === 修改 3: 返回所有模糊列表的集合 ===    
    return events, (blur_images_3, sharp_images, blur_images_5, blur_images_7, blur_images_9), image_timestamps, calib


def load_data_gen_coco_blur(path, only_event_time_range=False):
    print("Loading raw frames for custom blur synthesis from", path, "...")
    
    # 1. 获取所有原始帧的路径
    all_frame_paths = sorted(glob.glob(os.path.join(path, 'frames', '*.png')))
    
    # 2. 定义 GT 的选取逻辑
    step = 10
    start_idx = 9
    weights_3 = [0.2, 0.3, 0.5] 

    len_all_frames = len(all_frame_paths)
    target_count = len_all_frames // step
    print(f"Target aligned quantity from clean_gt_images: {target_count}")

    num_frames = len(all_frame_paths)
    gt_indices = list(range(start_idx, num_frames, step))
    gt_indices = gt_indices[:target_count]
    
    # === 修改 1: 初始化多个模糊程度的列表 ===
    blur_images_3 = []
    blur_images_5 = []
    blur_images_7 = []
    blur_images_9 = []
    sharp_images = []

    print(f"Synthesizing {len(gt_indices)} blur pairs (3, 5, 7, 9 frames)...")

    # 辅助读取函数，避免重复代码
    def read_frames(indices):
        return [cv2.imread(all_frame_paths[i], cv2.IMREAD_GRAYSCALE) for i in indices]

    for gt_idx in gt_indices:
        if gt_idx + 1 >= num_frames:
            continue

        # --- 基础 3帧模糊 (原有逻辑) ---
        if gt_idx < 2: 
            print(f"Skipping frame {gt_idx}, not enough history.")
            continue
            
        # 读取 idx-2, idx-1, idx
        indices_3 = [gt_idx - 2, gt_idx - 1, gt_idx]
        frames_3 = read_frames(indices_3)
        
        if any(img is None for img in frames_3):
            continue

        gt_img = frames_3[-1] # 当前帧即为 GT
        
        # 合成 3帧模糊
        blur_3 = synthesize_blur_weighted(frames_3, weights=weights_3)
        
        # === 修改 2: 新增 5, 7, 9 帧模糊逻辑 ===
        # 策略：如果历史帧不够，直接复用 blur_3
        
        # --- 5帧模糊 ---
        if gt_idx >= 4:
            # 读取 idx-4, idx-3 ...
            # 优化：复用已读取的 frames_3，只读额外的
            extra_indices = [gt_idx - 4, gt_idx - 3]
            extra_frames = read_frames(extra_indices)
            if any(img is None for img in extra_frames):
                blur_5 = blur_3 # 读取失败降级
            else:
                frames_5 = extra_frames + frames_3
                # 权重：线性递增 [1, 2, 3, 4, 5]
                blur_5 = synthesize_blur_weighted(frames_5, weights=list(range(1, 6)))
        else:
            blur_5 = blur_3 # 帧数不够，用3帧替代

        # --- 7帧模糊 ---
        if gt_idx >= 6:
            extra_indices = [gt_idx - 6, gt_idx - 5]
            extra_frames_7 = read_frames(extra_indices)
            if any(img is None for img in extra_frames_7):
                blur_7 = blur_3
            else:
                # 注意：这里需要 5帧所有的图 + 新增的2帧。
                # 为了代码简单，如果 blur_5 是降级的，这里也会受影响，所以重新构建完整列表最稳妥
                # 完整 indices: idx-6 ... idx
                full_indices_7 = list(range(gt_idx - 6, gt_idx + 1))
                frames_7 = read_frames(full_indices_7)
                blur_7 = synthesize_blur_weighted(frames_7, weights=list(range(1, 8)))
        else:
            blur_7 = blur_3

        # --- 9帧模糊 ---
        if gt_idx >= 8:
            full_indices_9 = list(range(gt_idx - 8, gt_idx + 1))
            frames_9 = read_frames(full_indices_9)
            if any(img is None for img in frames_9):
                blur_9 = blur_3
            else:
                blur_9 = synthesize_blur_weighted(frames_9, weights=list(range(1, 10)))
        else:
            blur_9 = blur_3

        # 存入列表
        blur_images_3.append(blur_3)
        blur_images_5.append(blur_5)
        blur_images_7.append(blur_7)
        blur_images_9.append(blur_9)
        sharp_images.append(gt_img)

    # 3. 处理 Events (逻辑保持不变)
    if only_event_time_range:
        npz_loaded = np.load(os.path.join(path, 'events', 'events.npz'))
        events = np.array([npz_loaded['events'][0, 4], npz_loaded['events'][-1, 4]])*1e-6
    else:
        npz_loaded = np.load(os.path.join(path, 'events', 'events.npz'))
        events_raw = np.array(npz_loaded['events'][:])
        x= events_raw[:, 1]
        y= events_raw[:, 2]
        p= events_raw[:, 3]
        ts= events_raw[:, 4]
        
        ts = ts * 1e-6  
        events = np.stack([ts, x, y, p], axis=1)
        events[:, -1] = np.clip(events[:, -1], 0, 1)

    # 4. 处理 Timestamps
    image_timestamps = np.loadtxt(os.path.join(path, 'frame_time.txt'), usecols=1) * 1e-6
    if "train" in path:
        image_timestamps = image_timestamps[start_idx::step]
    else:
        image_timestamps = image_timestamps[start_idx::step]
    image_timestamps = image_timestamps[:len(sharp_images)]    
    
    calib = np.array([1., 1., 0., 0., 0., 0., 0., 0., 0.]) 

    # === 修改 3: 返回所有模糊列表的集合 ===    
    return events, (blur_images_3, sharp_images, blur_images_5, blur_images_7, blur_images_9), image_timestamps, calib




# ----- Save time surfaces sparse and compressed ----- #
def save_ts_sparse(path, np_arr):
    # Convert to sparse matrix
    sparse_arr_list = [csr_matrix(np_arr[:, :, i]) for i in range(np_arr.shape[2])]

    # Save properties in lists
    data = np.array([csr.data for csr in sparse_arr_list], dtype=object)
    indices = np.array([csr.indices for csr in sparse_arr_list], dtype=object)
    indptr = np.array([csr.indptr for csr in sparse_arr_list], dtype=object)
    shapes = np.array([csr.shape for csr in sparse_arr_list], dtype=object)

    # Save
    np.savez_compressed(path, data=data, indices=indices, indptr=indptr, shapes=shapes)

# ----- Load time surfaces sparse and compressed ----- #
def load_ts_sparse(path):
    # Load
    loaded = np.load(path, allow_pickle=True)

    # Reconstruct CSR
    loaded_sparse_arr = [csr_matrix((loaded['data'][i], loaded['indices'][i], loaded['indptr'][i]), shape=loaded['shapes'][i]) for i in range(len(loaded['data']))]


    # Convert to np.ndarray
    loaded_sparse_arr = [arr.astype(float).toarray() for arr in loaded_sparse_arr]
    loaded_arr = np.stack(loaded_sparse_arr, axis=2)

    return loaded_arr

def synthesize_blur_weighted(images_list, weights=[0.2, 0.3, 0.5]):
    """辅助函数：加权合成模糊图"""
    # 转换为 float32 避免溢出
    stack = np.stack(images_list, axis=0).astype(np.float32)
    
    # 归一化权重
    w = np.array(weights, dtype=np.float32)
    w /= w.sum()
    
    # 加权求和
    h, w_img = stack.shape[1], stack.shape[2]
    accumulated = np.zeros((h, w_img), dtype=np.float32)
    for img, weight in zip(stack, w):
        accumulated += img * weight
    
    if len(images_list) >= 3:
        accumulated = cv2.GaussianBlur(accumulated, (3, 3), 0)
        
    return np.clip(accumulated, 0, 255).round().astype(np.uint8)