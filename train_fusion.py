#!/usr/bin/env python
import argparse
import numpy as np
import os
import sys
import time
import yaml

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from data.dataset import DataSplit, DatasetCollection, safe_collate_fn
from models.losses import super_event_loss, super_event_loss_fusion
from models.super_event_fusion import SuperEventFusion
from models.super_event import SuperEvent
from models.superpoint import SuperPointBatchNormed
from util.train_utils import list2device, val_loop_fusion
from util.eval_utils import fix_seed
# torch.autograd.set_detect_anomaly(True)

# Fix slurm output
sys.stdout.reconfigure(line_buffering=True, write_through=True)

# Fix seed for reproducibility
fix_seed()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/stage1.yaml", help="Parameter configuration.")
parser.add_argument("--model", default="", help="Pretrained model weights")
parser.add_argument("--slurm_id", default="")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
parser.add_argument("--gpu_number", type=int, default=0, help="Number of GPUs to use")
parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
parser.add_argument("--output_dir", default=".", help="Root for generated weights, logs and resume checkpoints.")
parser.add_argument("--max_train_batches", type=int, default=0, help="Optional short-run training cap; 0 uses the complete dataset.")
parser.add_argument("--max_val_batches", type=int, default=0, help="Optional validation cap; 0 uses the configured limit.")
parser.add_argument("--max_sequences_per_dataset", type=int, default=0, help="Optional data-loading cap for pipeline checks.")
parser.add_argument("--data_root", default="", help="Prepared training-data root; overrides train_data_path in the config.")
args = parser.parse_args()

# Load config
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
print("Loaded config from", args.config)
if "backbone" in config:
    # Load backbone config and add
    backbone_config_path = os.path.join(os.path.dirname(args.config), "backbones", config["backbone"] + ".yaml")
    if os.path.exists(backbone_config_path):
        with open(backbone_config_path, "r") as f:
            backbone_config = yaml.safe_load(f)
            print("Loaded backbone config from", backbone_config_path)
        config = config | backbone_config
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

if args.data_root:
    config["train_data_path"] = args.data_root

# Create experiment name: config_name + timestamp
config_name = os.path.splitext(os.path.basename(args.config))[0]
exp_name = config_name + "_" + config["train_type"] + "_" + time.strftime("%Y%m%d-%H%M%S")

# If slurm_id is specified, check if training with same ID exists
resume_training = False
if args.slurm_id:
    slurm_path = os.path.join(args.output_dir, "slurm", args.slurm_id)
    slurm_config_backup_path = os.path.join(slurm_path, "config.yaml")
    if os.path.isdir(slurm_path):
        # Load config and model
        with open(slurm_config_backup_path, "r") as f:
            config = yaml.safe_load(f)
        exp_name = config["exp_name"]
        resume_training = True

    else:
        os.makedirs(slurm_path)

        # Save config with model path
        config["exp_name"] = exp_name
        with open(slurm_config_backup_path, 'w') as f:
            yaml.dump(config, f)

if args.data_root:
    config["train_data_path"] = args.data_root

if args.max_sequences_per_dataset:
    config["max_sequences_per_dataset"] = args.max_sequences_per_dataset
if args.max_val_batches:
    config["max_num_validation_samples"] = args.max_val_batches * config["batch_size"]
print(f"Using config:\n\n{yaml.dump(config)}")

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

if config["train_type"] == "frame" or config["train_type"] == "event":
    config["input_channels"] = config["frame_input_channels"] if config["train_type"] == "frame" else config["event_input_channels"]
    model = SuperEvent(config).to(device)
else:
    model = SuperEventFusion(config).to(device)

sp_config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    }
}

if config.get("use_spbn", False):
    sp_model = SuperPointBatchNormed(**sp_config["superpoint"])#for feature loss
    spbn_path = config["frame_ckpt_pretrained"]
    sp_model.load_state_dict(torch.load(str(spbn_path), map_location='cpu', weights_only=True))
    sp_model.to(device)
else:
    raise ValueError("Only the MIT SuperPoint BN teacher is included; set use_spbn: true.")

sp_model.eval()
for param in sp_model.parameters():
    param.requires_grad = False

# Initialize logger and model_path
print("Experiment name:", exp_name)
model_weights_path = os.path.join(args.output_dir, "saved_models", exp_name + ".pth")
writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs", exp_name))
os.makedirs(os.path.dirname(model_weights_path), exist_ok=True)

# Load and Freeze backbone if specified

if config["train_type"] == "fusion":
    if "frame_ckpt_pretrained" in config and config["frame_ckpt_pretrained"] != "":
        frame_ckpt = torch.load(config["frame_ckpt_pretrained"], map_location='cpu', weights_only=True)
        frame_backbone_weights = {}
        if config["backbone"] not in ["sp-vgg", "sp", "vgg-mask-sp"] : 
            prefix = "backbone."
            for key, value in frame_ckpt.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):] # 去掉前缀
                    frame_backbone_weights[new_key] = value
        else: 
            for key, value in frame_ckpt.items():
                    frame_backbone_weights[key] = value
        
        model.frame_backbone.load_state_dict(frame_backbone_weights, strict=True)
        print(f"Loaded {len(frame_backbone_weights)} weights into model.frame_backbone.")

        if "frozen_backbone" in config and config["frozen_backbone"]:
            for param in model.frame_backbone.parameters():
                param.requires_grad = False
            print("Freezing frame backbone weights.")
        if config.get("frozen_backbone_normalization", False):
            model.frame_backbone.eval()
            print("Freezing frame backbone BatchNorm weights.")

    if "event_ckpt_pretrained" in config and config["event_ckpt_pretrained"] != "":            
        event_ckpt = torch.load(config["event_ckpt_pretrained"], map_location='cpu', weights_only=True)            
        event_backbone_weights = {}
        prefix = "backbone."
        for key, value in event_ckpt.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):] # 去掉前缀
                event_backbone_weights[new_key] = value
        model.event_backbone.load_state_dict(event_backbone_weights, strict=True)
        print(f"Loaded {len(event_backbone_weights)} weights into model.event_backbone.")

        if "frozen_backbone" in config and config["frozen_backbone"]:
            for param in model.event_backbone.parameters():
                param.requires_grad = False
            print("Freezing event backbone weights.")
        if config.get("frozen_backbone_normalization", False):
            model.event_backbone.eval()
            print("Freezing event backbone BatchNorm weights.")

        
    if "pretrained_fe_detector" in config and config["pretrained_fe_detector"]:
        event_detector_weights = {}
        event_descriptor_weights = {}
        prefix_det = "detector."
        prefix_desc = "descriptor."
        for key, value in event_ckpt.items():
            if key.startswith(prefix_det):
                new_key = key[len(prefix_det):] # 去掉前缀
                event_detector_weights[new_key] = value
            if key.startswith(prefix_desc):
                new_key = key[len(prefix_desc):] # 去掉前缀
                event_descriptor_weights[new_key] = value
            
        model.E_detector.load_state_dict(event_detector_weights, strict=True)
        print(f"Loaded {len(event_detector_weights)} weights into model.E_detector.")
        model.E_descriptor.load_state_dict(event_descriptor_weights, strict=True)
        print(f"Loaded {len(event_descriptor_weights)} weights into model.E_descriptor.")
    
        for param in model.E_detector.parameters():
            param.requires_grad = False
        for param in model.E_descriptor.parameters():
            param.requires_grad = False

        if config["backbone"] not in ["sp-vgg", "sp", "vgg-mask-sp"]:
            frame_detector_weights = {}
            frame_descriptor_weights = {}
            for key, value in frame_ckpt.items():
                if key.startswith(prefix_det):
                    new_key = key[len(prefix_det):] # 去掉前缀
                    frame_detector_weights[new_key] = value
                if key.startswith(prefix_desc):
                    new_key = key[len(prefix_desc):] # 去掉前缀
                    frame_descriptor_weights[new_key] = value

            model.F_detector.load_state_dict(frame_detector_weights, strict=True)
            print(f"Loaded {len(frame_detector_weights)} weights into model.F_detector.")
            model.F_descriptor.load_state_dict(frame_descriptor_weights, strict=True)
            print(f"Loaded {len(frame_descriptor_weights)} weights into model.F_descriptor.")

            for param in model.F_detector.parameters():
                param.requires_grad = False
            for param in model.F_descriptor.parameters():
                param.requires_grad = False

if config.get("use_spbn_decoder", False):
        print("Using SPBN weights to initialize detector and descriptor heads")
        sp_weights = torch.load(config["frame_ckpt_pretrained"])
        detector_state_dict = {}
        descriptor_state_dict = {}
        for key, value in sp_weights.items():
            if key.startswith("detector"):
                new_key = key.replace("detector.", "")
                detector_state_dict[new_key] = value
            elif key.startswith("descriptor"):
                new_key = key.replace("descriptor.", "")
                descriptor_state_dict[new_key] = value
        model.detector.layers.load_state_dict(detector_state_dict, strict=True)
        model.descriptor.layers.load_state_dict(descriptor_state_dict, strict=True)

# freeze decoder weights
if config["frozen_decoder"]:
    for param in model.detector.parameters():
        param.requires_grad = False
    for param in model.descriptor.parameters():
        param.requires_grad = False
    print("Frozen decoder weights")
if config.get("frozen_decoder_norm", False):
    model.detector.eval()
    model.descriptor.eval()
    print("Frozen decoder BatchNorm weights")
   

# Load pretrained weights if provided
if not args.model == "":
    model.load_state_dict(torch.load(args.model, weights_only=True))

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

# Some variables for training
best_val_loss = 1e6
best_epoch = 0
skip_to_idx = 0

if resume_training:
    train_checkpoint = torch.load(os.path.join(slurm_path, "train_checkpoint.pt"), weights_only=True)
    model.load_state_dict(train_checkpoint["model_state_dict"])
    optimizer.load_state_dict(train_checkpoint["optimizer_state_dict"])
    best_val_loss = train_checkpoint["best_val_loss"]
    best_epoch = train_checkpoint["best_epoch"]
    skip_to_idx = train_checkpoint["batch"] * config["batch_size"]

# Init data loaders
train_data = DatasetCollection(DataSplit.train, config, skip_to_idx=skip_to_idx, frames=True) #for fusion, frames must be True
val_data = DatasetCollection(DataSplit.val, config, frames=True)
if args.max_train_batches:
    train_data = Subset(train_data, range(min(len(train_data), args.max_train_batches * config["batch_size"])))
train_dataloader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=safe_collate_fn)
val_dataloader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=safe_collate_fn)

# Constants for training
epochs = config["epochs"]
num_samples = len(train_dataloader.dataset)
num_train_batches_before_validation = round(config["num_train_samples_before_validation"] / config["batch_size"])

for t in range(epochs):

    # Skip to current epoch when training is resumed
    if resume_training and train_checkpoint["epoch"] > t:
        continue

    print(f"\nEpoch {t}\n-------------------------------")
    train_total_loss = train_det_loss0 = train_det_loss1 = \
    train_fea_loss0 = train_fea_loss1 = train_desc_l1_loss = \
    train_keypoint_mse_loss = train_desc_loss = \
    train_pos_desc_loss = train_neg_desc_loss = \
    train_feature_lossE_hard = train_feature_lossF_hard = \
    train_det_loss_hard_E = train_det_loss_hard_F = \
    train_desc_l1_loss_hard = train_e_desc_loss = 0

    for batch, data in enumerate(train_dataloader):
        if data is None:
            print(f"Batch {batch} was skipped because all samples in it were invalid.")
            continue

        # Skip to current iteration when training is resumed
        if resume_training and train_checkpoint["batch"] > batch:
            continue
        else:
            resume_training = False

        # Set training mode
        model.train()
        if config["train_type"] == "fusion":
            if config.get("frozen_backbone_normalization", False):
                model.frame_backbone.eval()
                model.event_backbone.eval()

        if config.get("frozen_decoder_norm", False):
            model.detector.eval()
            model.descriptor.eval()

        # Channel dropout
        if config["ts_dropout"]:
            dropout_mask = [False]
            while not any(dropout_mask):  # prevent dropout on all channels
                dropout_mask = np.random.random_sample(size=5) < config["ts_dropout"]
            dropout_mask = np.hstack([dropout_mask, dropout_mask])
            data['ts0'][:, dropout_mask] = 0.
            data['ts1'][:, dropout_mask] = 0.

        # Compute prediction and loss
        data = list2device(data, device)
        ts0 = data['ts0']
        ts1 = data['ts1']
        labels0 = data['kp_map0']
        labels1 = data['kp_map1']
        frame0 = data['f0'].to(torch.float32).unsqueeze(1) 
        frame1 = data['f1'].to(torch.float32).unsqueeze(1)
        frame0_raw= data['f0_raw'].to(torch.float32).unsqueeze(1) 
        frame1_raw = data['f1_raw'].to(torch.float32).unsqueeze(1)

        mask_E0 = data['ts0_mask'].to(torch.float32)
        mask_E1 = data['ts1_mask'].to(torch.float32)
        mask_F0 = data['f0_mask'].to(torch.float32).unsqueeze(1)
        mask_F1 = data['f1_mask'].to(torch.float32).unsqueeze(1)
        mask_E0_exsits = data['ts0_exsit_mask']
        mask_E1_exsits = data['ts1_exsit_mask']

        # Forward pass
        if config["train_type"] == "fusion":
            if config["augmentation"]["frame"]["enable_in_train"] and config["augmentation"]["frame"]["enable"]:
                inputs_dict_0 = {
                    "frame_aug": frame0,
                    "label_map": labels0
                }
                inputs_dict_1 = {
                    "frame_aug": frame1,
                    "label_map": labels1
                }
                input_data0 = (frame0_raw/255, ts0, inputs_dict_0)
                input_data1 = (frame1_raw/255, ts1, inputs_dict_1)
            else:
                inputs_dict_0 = {
                    "label_map": labels0
                }
                inputs_dict_1 = {
                    "label_map": labels1
                }
                input_data0 = (frame0/255, ts0, inputs_dict_0)
                input_data1 = (frame1/255, ts1, inputs_dict_1)
        if config["train_type"] == "frame":
            input_data0 = (frame0/255, )
            input_data1 = (frame1/255, )
        if config["train_type"] == "event":
            input_data0 = (ts0, )
            input_data1 = (ts1, )
        results_ts0 = model(*input_data0)
        results_ts1 = model(*input_data1)

        with torch.no_grad():
            results_sp0 = sp_model(frame0_raw/255)
            results_sp1 = sp_model(frame1_raw/255)

        # det_loss0, det_loss1, desc_loss are only logged
        # loss, det_loss0, det_loss1, desc_loss, pos_desc_loss, neg_desc_loss = super_event_loss(results_ts0["logits"], results_ts1["logits"],
        #                                                                                        results_ts0["descriptors_raw"], results_ts1["descriptors_raw"],
        #                                                                                        labels0, labels1, config)

        if config.get("augmentation", {}).get("event", {}).get("enable_in_train", False):
            mask_E0 = results_ts0["mask_E"]
            mask_E1 = results_ts1["mask_E"]
        if config.get("augmentation", {}).get("frame", {}).get("enable_in_train", False):
            mask_F0 = results_ts0["mask_F"]
            mask_F1 = results_ts1["mask_F"]

        input_group = {
            "logits0": results_ts0["logits"],
            "logits1": results_ts1["logits"],
            "prob0": results_ts0["prob"],
            "prob1": results_ts1["prob"],
            "descriptors0": results_ts0["descriptors_raw"],
            "descriptors1": results_ts1["descriptors_raw"],
            "ground_truth_keypoint_map0": labels0,
            "ground_truth_keypoint_map1": labels1,
            "gt_descriptors0": results_sp0["descriptors_raw"],
            "gt_descriptors1": results_sp1["descriptors_raw"],
            "feature_0": results_ts0["features"],
            "feature_1": results_ts1["features"],
            "feature_sp0": results_sp0["features"][-1] if isinstance(results_sp0["features"], list) else results_sp0["features"],
            "feature_sp1": results_sp1["features"][-1] if isinstance(results_sp1["features"], list) else results_sp1["features"],
            "mask_E0": mask_E0,
            "mask_E1": mask_E1,
            "mask_F0": mask_F0,
            "mask_F1": mask_F1,
            "mask_E0_exsits": mask_E0_exsits.unsqueeze(1),
            "mask_E1_exsits": mask_E1_exsits.unsqueeze(1)
        }
        
        if config["train_type"] == "fusion":
            input_group.update({
            "feature_F0": results_ts0["f_features"],
            "feature_F1": results_ts1["f_features"],
            "feature_E0": results_ts0["e_features"],
            "feature_E1": results_ts1["e_features"]
            })
            if "pretrained_fe_detector" in config and config["pretrained_fe_detector"]:
                input_group.update({
                    "F_keypoint0": results_ts0["F_keypoint"],
                    "F_keypoint1": results_ts1["F_keypoint"],
                    "E_keypoint0": results_ts0["E_keypoint"],
                    "E_keypoint1": results_ts1["E_keypoint"],
                    "F_prob0": results_ts0["F_prob"],
                    "F_prob1": results_ts1["F_prob"],
                    "E_prob0": results_ts0["E_prob"],
                    "E_prob1": results_ts1["E_prob"],
                    "F_desc0": results_ts0["F_desc"],
                    "F_desc1": results_ts1["F_desc"],
                    "E_desc0": results_ts0["E_desc"],
                    "E_desc1": results_ts1["E_desc"]
                })

        if config["backbone"] == "vgg-mask-sp":
            input_group.update({
                "E_points_2d_0": results_ts0["E_points_2d"].unsqueeze(1),
                "E_points_2d_1": results_ts1["E_points_2d"].unsqueeze(1),
                "E_keypoints_nms_list_0": results_ts0["E_keypoints_nms_list"],
                "E_keypoints_nms_list_1": results_ts1["E_keypoints_nms_list"],
                "E_desc_nms_list_0": results_ts0["E_desc_nms_list"],
                "E_desc_nms_list_1": results_ts1["E_desc_nms_list"],
            })

        loss_dict = super_event_loss_fusion(input_group, config)

        loss = loss_dict["loss"]

        train_total_loss += loss
        train_det_loss0 += loss_dict["det_loss0"]
        train_det_loss1 += loss_dict["det_loss1"]
        train_desc_l1_loss += loss_dict["desc_l1_loss"]
        train_keypoint_mse_loss += loss_dict["keypoint_mse_loss"]
        train_desc_loss += loss_dict["desc_loss"]
        train_pos_desc_loss += loss_dict["pos_loss"]
        train_neg_desc_loss += loss_dict["neg_loss"]
        train_fea_loss0 += loss_dict["feature_loss0"]
        train_fea_loss1 += loss_dict["feature_loss1"]
        train_feature_lossE_hard += loss_dict["feature_lossE_hard"]
        train_feature_lossF_hard += loss_dict["feature_lossF_hard"]
        train_det_loss_hard_E += loss_dict["det_loss_hard_E"]
        train_det_loss_hard_F += loss_dict["det_loss_hard_F"]
        train_desc_l1_loss_hard += loss_dict["desc_l1_loss_hard"]
        train_e_desc_loss += loss_dict["e_desc_loss"]

        # Backpropagation
        accumulation_steps = config.get("gradient_accumulation_steps", 1)
        loss = loss / accumulation_steps
        loss.backward()

        if (batch + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get("max_grad_norm", 1.0))
            optimizer.step()
            optimizer.zero_grad()

        # Log train loss
        if batch % config["num_batches_train_logging"] == 0 and batch > 0:
            print(f"loss: {train_total_loss / config["num_batches_train_logging"]:>7f}  [{batch * config["batch_size"]:>5d}/{num_samples:>5d}]")

            # Tensorboard logging (start at 0)
            it = t * num_samples + config["batch_size"] * (batch - config["num_batches_train_logging"])
            writer.add_scalar("train/loss", train_total_loss / config["num_batches_train_logging"], it)
            writer.add_scalar("train/loss_det0", train_det_loss0 / config["num_batches_train_logging"], it)
            writer.add_scalar("train/loss_det1", train_det_loss1 / config["num_batches_train_logging"], it)
            writer.add_scalar("train/loss_desc_l1", train_desc_l1_loss / config["num_batches_train_logging"], it)
            writer.add_scalar("train/loss_keypoint_mse", train_keypoint_mse_loss / config["num_batches_train_logging"], it)
            writer.add_scalar("train/loss_desc", train_desc_loss / config["num_batches_train_logging"], it)
            writer.add_scalar("train/pos_desc_loss", train_pos_desc_loss / config["num_batches_train_logging"], it)
            writer.add_scalar("train/neg_desc_loss", train_neg_desc_loss / config["num_batches_train_logging"], it)
            writer.add_scalar("train/loss_fea0", train_fea_loss0 / config["num_batches_train_logging"], it)
            writer.add_scalar("train/loss_fea1", train_fea_loss1 / config["num_batches_train_logging"], it)
            writer.add_scalar("train/feature_lossE_hard", train_feature_lossE_hard / config["num_batches_train_logging"], it)
            writer.add_scalar("train/feature_lossF_hard", train_feature_lossF_hard / config["num_batches_train_logging"], it)
            writer.add_scalar("train/det_loss_hard_E", train_det_loss_hard_E / config["num_batches_train_logging"], it)
            writer.add_scalar("train/det_loss_hard_F", train_det_loss_hard_F / config["num_batches_train_logging"], it)
            writer.add_scalar("train/desc_l1_loss_hard", train_desc_l1_loss_hard / config["num_batches_train_logging"], it)
            writer.add_scalar("train/e_desc_loss", train_e_desc_loss / config["num_batches_train_logging"], it)

            # Reset train losses
            train_total_loss = train_det_loss0 = train_det_loss1 = \
            train_fea_loss0 = train_fea_loss1 = train_desc_l1_loss = \
            train_keypoint_mse_loss = train_desc_loss = \
            train_pos_desc_loss = train_neg_desc_loss = \
            train_feature_lossE_hard = train_feature_lossF_hard = \
            train_det_loss_hard_E = train_det_loss_hard_F = \
            train_desc_l1_loss_hard = train_e_desc_loss = 0

            # Save checkpoint to resume training if SLURM terminates job
            if args.slurm_id:
                batch_idx = batch + 1
                epoch_idx = t
                if batch_idx == len(train_dataloader):  # last batch in epoch
                    batch_idx = 0
                    epoch_idx = t + 1
                torch.save({
                    "epoch": epoch_idx,
                    "batch": batch_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch
                    }, os.path.join(slurm_path, "train_checkpoint.pt"))

        # Validate after specified number of iterations
        if batch % num_train_batches_before_validation == 0 and t + batch > 0:
            val_loss, val_det_loss0, val_det_loss1, val_fea_loss0, val_fea_loss1,\
            val_desc_l1_loss, val_keypoint_mse_loss, val_desc_loss, val_pos_desc_loss, val_neg_desc_loss, \
            val_feature_lossE_hard, val_feature_lossF_hard, val_det_loss_hard_E, val_det_loss_hard_F, \
            val_desc_l1_loss_hard, val_e_desc_loss \
            = val_loop_fusion(val_dataloader, model, sp_model, device, config)

            # Tensorboard logging (start at 0)
            it = t * num_samples + config["batch_size"] * (batch - num_train_batches_before_validation)
            writer.add_scalar("val/loss", val_loss, it)
            writer.add_scalar("val/loss_det0", val_det_loss0, it)
            writer.add_scalar("val/loss_det1", val_det_loss1, it)
            writer.add_scalar("val/loss_fea0", val_fea_loss0, it)
            writer.add_scalar("val/loss_fea1", val_fea_loss1, it)
            writer.add_scalar("val/loss_desc_l1", val_desc_l1_loss, it)
            writer.add_scalar("val/loss_keypoint_mse", val_keypoint_mse_loss, it)
            writer.add_scalar("val/loss_desc", val_desc_loss, it)
            writer.add_scalar("val/pos_loss_desc", val_pos_desc_loss, it)
            writer.add_scalar("val/neg_loss_desc", val_neg_desc_loss, it)
            writer.add_scalar("val/feature_lossE_hard", val_feature_lossE_hard, it)
            writer.add_scalar("val/feature_lossF_hard", val_feature_lossF_hard, it)
            writer.add_scalar("val/det_loss_hard_E", val_det_loss_hard_E, it)
            writer.add_scalar("val/det_loss_hard_F", val_det_loss_hard_F, it)
            writer.add_scalar("val/desc_l1_loss_hard", val_desc_l1_loss_hard, it)
            writer.add_scalar("val/e_desc_loss", val_e_desc_loss, it)


            # Check if this was the best validation loss so far
            if val_loss < best_val_loss:
                best_epoch = t + batch * config["batch_size"] / num_samples
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_weights_path)
                print(f"Model weights from epoch {best_epoch} saved in", model_weights_path)

model_final_weights_path = os.path.join(args.output_dir, "saved_models", exp_name + "_final.pth")
torch.save(model.state_dict(), model_final_weights_path)
writer.close()
print("Final model weights saved in", model_final_weights_path)
print("Training done.")
