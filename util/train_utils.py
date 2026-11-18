import numpy as np
from tqdm import tqdm

import torch

from models.losses import super_event_loss, super_event_loss_fusion, super_event_loss_fusion_deblur
from models.util import generate_gt_matches

from .eval_utils import extract_keypoints_and_descriptors_torch

def list2device(tensor_list, device):
    if type(tensor_list).__name__ == "list":
        tensor_list = [list2device(tensor, device) for tensor in tensor_list]
    elif type(tensor_list).__name__ == "dict":
        tensor_list = {key: list2device(tensor_list[key], device) for key in tensor_list}
    else:
        tensor_list = tensor_list.to(device)
    return tensor_list



def val_loop_fusion(dataloader, model, sp_model, device, config):
    """
        for frame-event fusion model
    """
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    # model.train()
    val_loss = val_det_loss0 = val_det_loss1 =\
    val_fea_loss0 = val_fea_loss1 = val_desc_l1_loss =\
    val_keypoint_mse_loss = val_desc_loss =\
    val_pos_desc_loss = val_neg_desc_loss =\
    val_feature_lossE_hard = val_feature_lossF_hard =\
    val_det_loss_hard_E = val_det_loss_hard_F =\
    val_desc_l1_loss_hard = val_e_desc_loss = 0.

    num_batches = min(round(config["max_num_validation_samples"] / config["batch_size"]), int(len(dataloader)))

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    processed_batches = 0
    with torch.no_grad():
        for batch, data in enumerate(tqdm(dataloader, total=num_batches)):
            if data is None:
                print(f"Batch {batch} was skipped because all samples in it were invalid.")
                continue

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

            val_loss += loss
            val_det_loss0 += loss_dict["det_loss0"]
            val_det_loss1 += loss_dict["det_loss1"]
            val_desc_l1_loss += loss_dict["desc_l1_loss"]
            val_keypoint_mse_loss += loss_dict["keypoint_mse_loss"]
            val_desc_loss += loss_dict["desc_loss"]
            val_pos_desc_loss += loss_dict["pos_loss"]
            val_neg_desc_loss += loss_dict["neg_loss"]
            val_fea_loss0 += loss_dict["feature_loss0"]
            val_fea_loss1 += loss_dict["feature_loss1"]
            val_feature_lossE_hard += loss_dict["feature_lossE_hard"]
            val_feature_lossF_hard += loss_dict["feature_lossF_hard"]
            val_det_loss_hard_E += loss_dict["det_loss_hard_E"]
            val_det_loss_hard_F += loss_dict["det_loss_hard_F"]
            val_desc_l1_loss_hard += loss_dict["desc_l1_loss_hard"]
            val_e_desc_loss += loss_dict["e_desc_loss"]

            # Only use predefined number of validation batches
            processed_batches += 1
            if processed_batches >= num_batches:
                break

    if processed_batches == 0:
        raise RuntimeError("Validation did not receive any valid batches.")
    val_loss /= processed_batches
    print(f"Val Loss: {val_loss:>8f}")
    return val_loss, \
           val_det_loss0 / processed_batches, \
           val_det_loss1 / processed_batches, \
           val_fea_loss0 / processed_batches, \
           val_fea_loss1 / processed_batches, \
           val_desc_l1_loss / processed_batches, \
           val_keypoint_mse_loss / processed_batches, \
           val_desc_loss / processed_batches, \
           val_pos_desc_loss / processed_batches, \
           val_neg_desc_loss / processed_batches, \
           val_feature_lossE_hard / processed_batches, \
           val_feature_lossF_hard / processed_batches, \
           val_det_loss_hard_E / processed_batches, \
           val_det_loss_hard_F / processed_batches, \
           val_desc_l1_loss_hard / processed_batches, \
           val_e_desc_loss / processed_batches



def val_loop_fusion_deblur(dataloader, model, sp_model, device, config):
    """
        for frame-event fusion model
    """
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    # model.train()
    val_loss = val_det_loss0 = val_det_loss1 =\
    val_fea_loss = val_desc_loss =\
    val_l1_sparsity_loss =\
    val_pos_desc_loss = val_neg_desc_loss = \
    val_det_logits_loss = val_desc_fea_loss = 0.

    num_batches = min(round(config["max_num_validation_samples"] / config["batch_size"]), int(len(dataloader)))

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    processed_batches = 0
    with torch.no_grad():
        for batch, data in enumerate(tqdm(dataloader, total=num_batches)):
            if data is None:
                print(f"Batch {batch} was skipped because all samples in it were invalid.")
                continue

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

            mask_E0_exsits = data['ts0_exsit_mask']
            mask_E1_exsits = data['ts1_exsit_mask']

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

            results_sp0 = sp_model(frame0_raw/255)
            results_sp1 = sp_model(frame1_raw/255)

            # det_loss0, det_loss1, desc_loss are only logged
            # loss, det_loss0, det_loss1, desc_loss, pos_desc_loss, neg_desc_loss = super_event_loss(results_ts0["logits"], results_ts1["logits"],
            #                                                                                        results_ts0["descriptors_raw"], results_ts1["descriptors_raw"],
            #                                                                                        labels0, labels1, config)
            
            input_group = {
                "ground_truth_keypoint_map0": labels0,
                "ground_truth_keypoint_map1": labels1,
                "logits0": results_ts0["logits"],
                "logits1": results_ts1["logits"],
                "prob0": results_ts0["prob"],
                "prob1": results_ts1["prob"],
                "descriptors0": results_ts0["descriptors_raw"],
                "descriptors1": results_ts1["descriptors_raw"],
                "descriptors0_expanded": results_ts0["descriptors"],
                "descriptors1_expanded": results_ts1["descriptors"],
                "feature_0": results_ts0["features"],
                "feature_1": results_ts1["features"],
                "feature_e0": results_ts0["e_features"],
                "feature_e1": results_ts1["e_features"], 
                "feature_f0": results_ts0["f_features"],
                "feature_f1": results_ts1["f_features"],

                "gt_logits0": results_sp0["logits"],
                "gt_logits1": results_sp1["logits"],
                "gt_descriptors0": results_sp0["descriptors_raw"],
                "gt_descriptors1": results_sp1["descriptors_raw"],
                "gt_descriptors0_expanded": results_sp0["descriptors"],
                "gt_descriptors1_expanded": results_sp1["descriptors"],
                "feature_sp0": results_sp0["features"][-1] if isinstance(results_sp0["features"], list) else results_sp0["features"],
                "feature_sp1": results_sp1["features"][-1] if isinstance(results_sp1["features"], list) else results_sp1["features"],

                "mask_E0_exsits": mask_E0_exsits.unsqueeze(1),
                "mask_E1_exsits": mask_E1_exsits.unsqueeze(1)
            }

            if config["backbone"] == "vgg-mask-sp":
                input_group.update({
                    "E_points_2d_0": results_ts0["E_points_2d"].unsqueeze(1),
                    "E_points_2d_1": results_ts1["E_points_2d"].unsqueeze(1),
                })
            
            if config.get("clear_blur_matching", {}).get("enable", False):
                input_group.update({
                    "pixel_mapping": data["pixel_mapping"],
                })
            
            if config.get("use_seprate_decoder", False):
                input_group.update({
                    "feature_0_detector": results_ts0["features_detector"],
                    "feature_1_detector": results_ts1["features_detector"],
                    "feature_0_desc": results_ts0["features_descriptor"],
                    "feature_1_desc": results_ts1["features_descriptor"],
                })
        
            loss_dict = super_event_loss_fusion_deblur(input_group, config)

            loss = loss_dict["loss"]

            val_loss += loss
            val_det_loss0 += loss_dict["det_loss0"]
            val_det_loss1 += loss_dict["det_loss1"]
            val_desc_loss += loss_dict["desc_loss"]
            val_pos_desc_loss += loss_dict["pos_loss"]
            val_neg_desc_loss += loss_dict["neg_loss"]
            val_fea_loss += loss_dict["feature_loss"]
            val_l1_sparsity_loss += loss_dict["desc_l1_sparsity_loss"]
            val_det_logits_loss += loss_dict["det_logits_loss"]
            val_desc_fea_loss += loss_dict["desc_fea_loss"]

            # Only use predefined number of validation batches
            processed_batches += 1
            if processed_batches >= num_batches:
                break

    if processed_batches == 0:
        raise RuntimeError("Validation did not receive any valid batches.")
    val_loss /= processed_batches
    print(f"Val Loss: {val_loss:>8f}")
    return val_loss, \
           val_det_loss0 / processed_batches, \
           val_det_loss1 / processed_batches, \
           val_fea_loss / processed_batches, \
           val_l1_sparsity_loss / processed_batches, \
           val_desc_loss / processed_batches, \
           val_pos_desc_loss / processed_batches, \
           val_neg_desc_loss / processed_batches, \
           val_det_logits_loss / processed_batches, \
           val_desc_fea_loss / processed_batches

