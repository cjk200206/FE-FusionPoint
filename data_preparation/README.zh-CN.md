# 数据准备

[English](README.md) | 简体中文

在仓库根目录运行，依次完成原始数据准备、CPU 并行预处理、GPU 伪标签和数据划分。

## 1. 原始数据

按 `dataset/<数据集名>/<序列名>/` 放置序列，数据集名使用小写。

| 数据集 | 每个序列的输入 |
|---|---|
| `mvsec` | 包含事件、图像和时间戳的 `*_data.hdf5` |
| `vivid`、`griffin` | `images/`、`images.txt`、`events.txt` |
| `ddd20` | 已导出的 `*.hdf5.exported.hdf5`，包含 `event`、`frame`、`frame_ts` |
| `fpv` | `images.txt` 及其中记录的图像、`events.txt` |
| `uevd` | `frames/`、`clean_gt_images/`、`events/events.npz`、`frame_time.txt` |
| `gen_coco`、`gen_coco_1` | 下节生成的 `frames/`、`events/events.npz`、`frame_time.txt` |

ROS 数据需安装 `rosbags`，在 `convert_rosbags.py` 中设置 `BAG_ROOT`、`OUTPUT_DIR` 和 `FILE_NAMES`，运行 `python data_preparation/convert_rosbags.py`。默认输出到 `dataset/griffin`；Vivid 更换为对应路径和序列列表。

## 2. COCO 合成

准备 `dataset/coco/train2014/` 和 `dataset/coco/annotations/instances_train2014.json`。筛选图片需 `pycocotools`，GPU 事件模拟需 `numba` 和 OpenEB/Metavision 提供的 `metavision_core_ml`。

```bash
python data_preparation/select_coco_images.py
CUDA_VISIBLE_DEVICES=0 python data_preparation/gen_coco_events.py \
  --output_dir dataset/gen_coco --num_samples 1500 --mode mixed
CUDA_VISIBLE_DEVICES=0 python data_preparation/gen_coco_events.py \
  --output_dir dataset/gen_coco_1 --num_samples 200 --mode rotation
```

## 3. CPU 并行预处理

对每个数据集运行一次，`-j` 指定并发序列数，默认 32：

```bash
bash data_preparation/prepare_training_data_parallel.sh \
  -d dataset/uevd -t dataset/SuperEvent_data -j 8
```

生成图像和 10 通道 MCTS；UEVD、合成 COCO 同时生成清晰帧、多级模糊图及配套 MCTS。去畸变时加 `-u`，并在每个原始序列中放置 `calib.txt`。

## 4. GPU 串行伪标签

获取 SuperGlue，使用与 CPU 阶段相同的 `-d`、`-t` 路径：

```bash
git submodule update --init -- data_preparation/SuperGluePretrainedNetwork
CUDA_VISIBLE_DEVICES=0 bash data_preparation/prepare_training_data.sh \
  -d dataset/uevd -t dataset/SuperEvent_data
```

逐序列生成 `sg_matches/`；UEVD、合成 COCO 使用清晰帧。对其他数据集重复第 3、4 步。

## 5. 划分并训练

所有数据集准备完成后运行：

```bash
python data_preparation/split_data.py dataset/SuperEvent_data config/stage1.yaml
```

按配置将序列移到 `train/`、`val/`、`test/`，得到 `SuperEvent_data/<划分>/<数据集>/<序列>/`。两阶段训练共用此目录，通过 `--data_root` 指定；训练命令见[项目说明](../README.zh-CN.md#训练)。

## 6. 测评数据

ECD 每个序列包含 `images/`、`images.txt`、`groundtruth.txt`，将 `events.txt` 转为 `events.npy`：

```bash
python data_preparation/txt2npy.py /path/to/EC/boxes_6dof/events.txt
```

对其他 ECD 序列重复此命令。MVSEC 每个序列准备 `<序列名>_data.hdf5` 和 `<序列名>_gt.hdf5`。

评测脚本在线生成 MCTS，使用随附的固定帧对和标定。运行命令见[项目说明](../README.zh-CN.md#评测)。
