# Data preparation

English | [简体中文](README.zh-CN.md)

Run all commands from the repository root: prepare raw data, preprocess sequences on CPU, generate pseudo-labels on GPU, then split the data.

## 1. Raw data

Place sequences under `dataset/<dataset>/<sequence>/`, using lowercase dataset names.

| Dataset | Input per sequence |
|---|---|
| `mvsec` | `*_data.hdf5` containing events, images, and timestamps |
| `vivid`, `griffin` | `images/`, `images.txt`, `events.txt` |
| `ddd20` | Exported `*.hdf5.exported.hdf5` containing `event`, `frame`, and `frame_ts` |
| `fpv` | `images.txt`, the images it references, and `events.txt` |
| `uevd` | `frames/`, `clean_gt_images/`, `events/events.npz`, `frame_time.txt` |
| `gen_coco`, `gen_coco_1` | `frames/`, `events/events.npz`, and `frame_time.txt` generated below |

For ROS data, install `rosbags`, set `BAG_ROOT`, `OUTPUT_DIR`, and `FILE_NAMES` in `convert_rosbags.py`, then run `python data_preparation/convert_rosbags.py`. The default output is `dataset/griffin`; use the corresponding paths and sequence list for Vivid.

## 2. COCO synthesis

Prepare `dataset/coco/train2014/` and `dataset/coco/annotations/instances_train2014.json`. Image selection requires `pycocotools`; GPU event simulation requires `numba` and `metavision_core_ml` from OpenEB/Metavision.

```bash
python data_preparation/select_coco_images.py
CUDA_VISIBLE_DEVICES=0 python data_preparation/gen_coco_events.py \
  --output_dir dataset/gen_coco --num_samples 1500 --mode mixed
CUDA_VISIBLE_DEVICES=0 python data_preparation/gen_coco_events.py \
  --output_dir dataset/gen_coco_1 --num_samples 200 --mode rotation
```

## 3. Parallel CPU preprocessing

Run once per dataset. `-j` sets the number of concurrent sequences (default: 32).

```bash
bash data_preparation/prepare_training_data_parallel.sh \
  -d dataset/uevd -t dataset/SuperEvent_data -j 8
```

This generates images and 10-channel MCTS. UEVD and synthetic COCO also produce sharp frames, multiple blur levels, and the corresponding MCTS. For undistortion, add `-u` and place `calib.txt` in each raw sequence directory.

## 4. Sequential GPU pseudo-label generation

Obtain SuperGlue and use the same `-d` and `-t` paths as in the CPU stage:

```bash
git submodule update --init -- data_preparation/SuperGluePretrainedNetwork
CUDA_VISIBLE_DEVICES=0 bash data_preparation/prepare_training_data.sh \
  -d dataset/uevd -t dataset/SuperEvent_data
```

This generates `sg_matches/` one sequence at a time, using sharp frames for UEVD and synthetic COCO. Repeat steps 3 and 4 for the other datasets.

## 5. Split and train

After preparing all datasets, run:

```bash
python data_preparation/split_data.py dataset/SuperEvent_data config/stage1.yaml
```

Sequences are moved into `train/`, `val/`, and `test/` according to the configuration, producing `SuperEvent_data/<split>/<dataset>/<sequence>/`. Both training stages share this directory, passed through `--data_root`. See [Training](../README.md#training).

## 6. Evaluation data

Each ECD sequence contains `images/`, `images.txt`, and `groundtruth.txt`. Convert `events.txt` to `events.npy`:

```bash
python data_preparation/txt2npy.py /path/to/EC/boxes_6dof/events.txt
```

Repeat for the other ECD sequences. Each MVSEC sequence needs `<sequence>_data.hdf5` and `<sequence>_gt.hdf5`.

Evaluation generates MCTS online and uses the supplied fixed pairs and calibration. See [Evaluation](../README.md#evaluation) for commands.
