import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

from models.backbones.vgg import VggBackbone, VggBackbone_Upsample, SuperPointBackboneBatchNormed, SPBNBackboneWithEvent
from models.backbones.maxvit import MaxViTBackbone
from models.backbones.maxvit_backbone.yolo_pafpn import YOLOPAFPN
from models.heads import DetectorHead, DetectorHeadFullRes, DescriptorHead, DetectorHeadFusion, DescriptorHeadFusion
from models.backbones.maxvit_backbone.fusion_blocks import FusionBlock

from models.util import fast_nms, create_occlusion_mask_FAST, batch_mutual_nearest_neighbor
from util.eval_utils import extract_keypoints_and_descriptors_torch

class SuperEventFusion(nn.Module):
    def __init__(self, config, tracing=False):
        super().__init__()
        self.backbone_type = config["backbone"]
        self.tracing = tracing
        self.fusion_decoder = False
        self.use_frame_only_for_descriptor = config.get("use_frame_only_for_descriptors", False)
        self.use_seprate_decoder = config.get("use_seprate_decoder", False)
        self.frame_aug = config["augmentation"]["frame"]
        self.event_aug = config["augmentation"]["event"]
        self.config = config

        self.fpn = None

        if config["backbone"] == "vgg":
            self.frame_backbone = VggBackbone(input_channels=config["frame_input_channels"], output_channels=config["feature_channels"],norm=config.get("vgg_norm","bn"))
            self.event_backbone = VggBackbone(input_channels=config["event_input_channels"], output_channels=config["feature_channels"],norm=config.get("vgg_norm","bn"))
            self.fusion = FusionBlock(config["fusion_config"])
            
        elif config["backbone"] in ["sp-vgg", "vgg-mask-sp"]:
            if config.get("use_spbn", False):
                print("Using SuperPoint Backbone with BatchNorm for frame backbone")
                self.frame_backbone = SuperPointBackboneBatchNormed(config)
            else:
                raise ValueError("The non-BN SuperPoint backbone is not included; set use_spbn: true.")
            self.event_backbone = VggBackbone(input_channels=config["event_input_channels"], output_channels=config["feature_channels"], norm=config.get("vgg_norm","bn"))
            self.fusion = FusionBlock(config["fusion_config"])

            if config["fusion_config"]["is_FPN"]:
                print("Using FPN in fusion block")
                self.fpn = YOLOPAFPN(config,
                                     depth=config["fusion_config"]["fpn"]["depth"], 
                                     in_stages=np.array(config["fusion_config"]["fpn"]["in_stages"])[-4:].tolist(),
                                     in_channels=(config["fusion_config"]["embed_dim"] * np.array(config["fusion_config"]["dim_multiplier"])[-4:]).tolist(),
                                     depthwise=config["fusion_config"]["fpn"]["depthwise"],
                                     act=config["fusion_config"]["fpn"]["act"])
                
        elif config["backbone"] == "sp-vgg-deblur":
            if config.get("use_event_injection", False):
                print("Using SuperPoint Backbone with Event Injection for frame backbone")
                self.frame_backbone = SPBNBackboneWithEvent(config)
            else:
                self.frame_backbone = SuperPointBackboneBatchNormed(config)
            self.event_backbone = VggBackbone(input_channels=config["event_input_channels"], output_channels=config["feature_channels"], 
                                              norm=config.get("vgg_norm","bn"), enable_dog=config.get("use_event_enable_dog", False),
                                              enable_selector=config.get("use_event_enable_selector", False))
            if config.get("use_seprate_decoder", False):
                config["fusion_config"]["use_seprate_decoder"] = True
                print("Using separate decoder for fusion block")
            self.fusion = FusionBlock(config["fusion_config"])

            if config["fusion_config"]["is_FPN"]:
                print("Using FPN in fusion block")
                self.fpn = YOLOPAFPN(config,
                                     depth=config["fusion_config"]["fpn"]["depth"], 
                                     in_stages=np.array(config["fusion_config"]["fpn"]["in_stages"])[-4:].tolist(),
                                     in_channels=(config["fusion_config"]["embed_dim"] * np.array(config["fusion_config"]["dim_multiplier"])[-4:]).tolist(),
                                     depthwise=config["fusion_config"]["fpn"]["depthwise"],
                                     act=config["fusion_config"]["fpn"]["act"])


        elif config["backbone"] == "maxvit":
            config["backbone_config"]["input_channels"] = config["frame_input_channels"]
            self.frame_backbone = MaxViTBackbone(config["backbone_config"])

            config["backbone_config"]["input_channels"] = config["event_input_channels"]
            self.event_backbone = MaxViTBackbone(config["backbone_config"])
            
            self.fusion = FusionBlock(config["fusion_config"])

            self.fpn = YOLOPAFPN(config,
                                 depth=config["backbone_config"]["fpn"]["depth"], 
                                 in_stages=np.array(config["backbone_config"]["fpn"]["in_stages"])[-4:].tolist(),
                                 in_channels=(config["backbone_config"]["embed_dim"] * np.array(config["backbone_config"]["dim_multiplier"])[-4:]).tolist(),
                                 depthwise=config["backbone_config"]["fpn"]["depthwise"],
                                 act=config["backbone_config"]["fpn"]["act"])
        
        else:
            raise NotImplementedError("Backbone", config["backbone"], " is not supported.")
        
        self.pre_trained_detector = False
        if "pretrained_fe_detector" in config and config["pretrained_fe_detector"]:
            self.pre_trained_detector = True
            self.E_detector = DetectorHead(input_channels=config["backbone_output_channels"], grid_size=config["grid_size"], config=config)
            self.E_descriptor = DescriptorHead(input_channels=config["backbone_output_channels"], grid_size=config["grid_size"], descriptor_size=config["descriptor_size"], config=config)        
            if config["backbone"]!="sp-vgg":
                self.F_detector = DetectorHead(input_channels=config["backbone_output_channels"], grid_size=config["grid_size"], config=config)
                self.F_descriptor = DescriptorHead(input_channels=config["backbone_output_channels"], grid_size=config["grid_size"], descriptor_size=config["descriptor_size"], config=config)

        if "pretrained_fe_detector" in config and config["pretrained_fe_detector"] and config["fusion_decoder"]:
            self.fusion_decoder = True
            self.detector = DetectorHeadFusion(input_channels=config["backbone_output_channels"], grid_size=config["grid_size"], config=config)
            self.descriptor = DescriptorHeadFusion(input_channels=config["backbone_output_channels"], grid_size=config["grid_size"], descriptor_size=config["descriptor_size"], config=config)
            
        else:
            self.detector = DetectorHead(input_channels=config["backbone_output_channels"], grid_size=config["grid_size"], config=config, use_spbn=config.get("use_spbn_decoder", False))
            self.descriptor = DescriptorHead(input_channels=config["backbone_output_channels"], grid_size=config["grid_size"], descriptor_size=config["descriptor_size"], config=config, use_spbn=config.get("use_spbn_decoder", False))




    def forward(self, x_frame, x_event, x_inputs:dict=None):
        fusion_input = {}
        if self.backbone_type == "vgg":
            f_features = self.frame_backbone(x_frame)[0][-1]
            e_features = self.event_backbone(x_event)[0][-1]
        elif self.backbone_type == "sp-vgg":
            sp_out = self.frame_backbone(x_frame)
            f_features = sp_out['features']
            e_features = self.event_backbone(x_event)[0][-1]

            if self.pre_trained_detector:
                F_keypoint = sp_out["logits"]
                F_prob = torch.nn.functional.softmax(F_keypoint, dim=1)
                F_prob = F_prob[:,:-1,:,:]
                F_prob = torch.nn.functional.pixel_shuffle(F_prob, 8)
                F_prob = torch.squeeze(F_prob, dim=1)
                E_keypoint, E_prob = self.E_detector(e_features)

                F_desc = sp_out["descriptors"]
                E_desc, _ = self.E_descriptor(e_features)
                E_desc = F.normalize(E_desc, p=2, dim=1)
                
                fusion_input["F_keypoint"] = F_keypoint
                fusion_input["E_keypoint"] = E_keypoint
                fusion_input["F_desc"] = F_desc
                fusion_input["E_desc"] = E_desc

        elif self.backbone_type == "vgg-mask-sp":
            
            with torch.no_grad():
                e_features_raw = self.event_backbone(x_event)[0][-1]
                E_keypoint_raw, E_prob_raw = self.E_detector(e_features_raw)

                E_desc_raw, E_desc_interpolated = self.E_descriptor(e_features_raw)
                E_desc_raw = F.normalize(E_desc_raw, p=2, dim=1)

            
                mask_F = torch.ones_like(x_frame)
                mask_E = torch.ones_like(x_event[:,0:1,:,:])

                
                config_nms = {
                    "detection_threshold": 0.01,
                    "nms_box_size": 5
                }
                E_points_nms, E_prob_nms = fast_nms(E_prob_raw, config_nms, top_k=256) 

                E_points = torch.zeros_like(E_prob_raw)
                E_keypoints_nms_list = []
                E_desc_nms_list = []
                for b in range(E_points.shape[0]):
                    E_points[b][E_points_nms[b][:,0], E_points_nms[b][:,1]] = 1.0

                    E_keypoints_nms, E_desc_nms = extract_keypoints_and_descriptors_torch(E_points_nms[b], E_desc_interpolated[b], resize_factor=1.0)
                    
                    E_keypoints_nms_list.append(E_keypoints_nms)
                    E_desc_nms_list.append(E_desc_nms)

                

                corrupt_prob = self.frame_aug["in_train_dropout_p"]
                if torch.rand(1).item() < corrupt_prob:
                    if self.frame_aug["enable_in_train"]:
                        # frame aug 
                        if x_inputs is not None and "label_map" in x_inputs:
                            gt_label = x_inputs["label_map"].unsqueeze(1)
                            mask_F, mask_value = create_occlusion_mask_FAST(E_points.unsqueeze(1), 4, 80, gt_keypoint_map_batch=gt_label)
                        else:
                            mask_F, mask_value = create_occlusion_mask_FAST(E_points.unsqueeze(1), 4, 80)

                        if self.frame_aug["enable"]:
                            x_aug = x_inputs["frame_aug"]
                        else:
                            x_aug = mask_value   
                        # x_frame = x_frame * mask_F + mask_value
                        x_frame = x_frame * mask_F + (1 - mask_F) * x_aug

                    if self.event_aug["enable_in_train"]:
                        # event aug
                        E_points_masked = E_points * mask_F.squeeze(1)
                        if x_inputs is not None and "label_map" in x_inputs:
                            gt_label = x_inputs["label_map"].unsqueeze(1)
                            mask_E, _ = create_occlusion_mask_FAST(E_points_masked.unsqueeze(1), 1, 60, gt_keypoint_map_batch=gt_label)
                        else:
                            mask_E, _ = create_occlusion_mask_FAST(E_points_masked.unsqueeze(1), 1, 60)
                        x_event = x_event * mask_E

            #####new
            e_features = self.event_backbone(x_event)[0]
            E_keypoint, E_prob = self.E_detector(e_features[-1])

            E_desc, _ = self.E_descriptor(e_features[-1])
            E_desc = F.normalize(E_desc, p=2, dim=1)
            ####

            sp_out = self.frame_backbone(x_frame)
            f_features = sp_out['features']

            F_keypoint = sp_out["logits"]
            F_prob = torch.nn.functional.softmax(F_keypoint, dim=1)
            F_prob = F_prob[:,:-1,:,:]
            F_prob = torch.nn.functional.pixel_shuffle(F_prob, 8)
            F_prob = torch.squeeze(F_prob, dim=1)
            F_desc = sp_out["descriptors_raw"]
            
            fusion_input["F_keypoint"] = F_keypoint
            fusion_input["E_keypoint"] = E_keypoint
            fusion_input["F_desc"] = F_desc
            fusion_input["E_desc"] = E_desc

        elif self.backbone_type == "sp-vgg-deblur":
            e_features = self.event_backbone(x_event)[0]

            sp_out = self.frame_backbone(x_frame, e_features)
            f_features = sp_out['features']


        elif self.backbone_type == "maxvit":
            f_features = self.frame_backbone(x_frame)
            e_features = self.event_backbone(x_event)

        fusion_input["frame_features"] = f_features
        fusion_input["event_features"] = e_features

        if self.use_seprate_decoder:
            features_detector, features_descriptor = self.fusion(fusion_input)
            features = features_detector
        else:
            features = self.fusion(fusion_input)

        if self.fpn is not None:
            features = self.fpn(features)
            
        fusion_input["fusion_feature"] = features

        if self.fusion_decoder:
            logits, prob = self.detector(fusion_input)
            descriptors_raw, descriptors = self.descriptor(fusion_input)
        elif self.use_frame_only_for_descriptor:
            # logits, prob = self.detector(f_features[-1])
            # descriptors_raw, descriptors = self.descriptor(features)
            logits, prob = self.detector(features)
            descriptors_raw, descriptors = self.descriptor(f_features[-1])
        elif self.use_seprate_decoder:
            logits, prob = self.detector(features_detector)
            descriptors_raw, descriptors = self.descriptor(features_descriptor)
        else:
            logits, prob = self.detector(features)
            descriptors_raw, descriptors = self.descriptor(features)

        output = {"logits": logits,
                    "prob": prob, 
                    "descriptors_raw": descriptors_raw, # without normalization
                    "descriptors": descriptors,
                    "features": features,
                    "f_features": f_features[-1],
                    "e_features": e_features[-1]}
        
        if self.pre_trained_detector:
            output.update({
                "F_keypoint": F_keypoint,
                "E_keypoint": E_keypoint,
                "F_prob": F_prob,
                "E_prob": E_prob,
                "F_desc": F_desc,
                "E_desc": E_desc,
                "mask_F": mask_F.detach(),
                "mask_E": mask_E.detach(),
                "frame_augmented": x_frame,
                "event_augmented": x_event,
            })

        if self.backbone_type == "vgg-mask-sp":
            output.update({
                "E_points_2d": E_points.detach(),
                "E_keypoints_nms_list": E_keypoints_nms_list,
                "E_desc_nms_list": E_desc_nms_list
            })

        if self.use_seprate_decoder:
            output["features_detector"] = features_detector
            output["features_descriptor"] = features_descriptor

        if self.tracing:
            return prob, descriptors
        else:
            return output
    
class SuperEventFullRes(nn.Module):
    def __init__(self, config, tracing=False):
        super().__init__()
        self.config = config
        self.tracing = tracing

        if config["backbone"] == "vgg":
            self.backbone_down = VggBackbone(input_channels=config["input_channels"], output_channels=config["feature_channels"], return_maxpool_indeces=True)
            self.backbone_up = VggBackbone_Upsample(input_channels=config["feature_channels"], output_channels=config["backbone_output_channels"])
        elif config["backbone"] == "maxvit":
            self.backbone = MaxViTBackbone(config["backbone_config"])
            self.fpn = YOLOPAFPN(config,
                                 depth=config["backbone_config"]["fpn"]["depth"], 
                                 in_stages=config["backbone_config"]["fpn"]["in_stages"],
                                 in_channels=config["backbone_config"]["embed_dim"] * np.array(config["backbone_config"]["dim_multiplier"]),
                                 depthwise=config["backbone_config"]["fpn"]["depthwise"],
                                 act=config["backbone_config"]["fpn"]["act"])
        else:
            raise NotImplementedError("Backbone", config["backbone"], " is not supported.")

        self.detector = DetectorHeadFullRes(input_channels=config["backbone_output_channels"], config=config)
        self.descriptor = DescriptorHead(input_channels=config["backbone_output_channels"], grid_size=1, descriptor_size=config["descriptor_size"], interpolate=False, config=config)

    def forward(self, x):
        if self.backbone_type == "vgg":
            features_compressed, maxpool_indeces = self.backbone_down(x)
            features = self.backbone_up(features_compressed, maxpool_indeces)
        elif self.backbone_type == "maxvit":
            backbone_features = self.backbone(x)
            features = self.fpn(backbone_features)

        logits, prob = self.detector(features)
        descriptors_raw, descriptors = self.descriptor(features)

        if self.tracing:
            return prob, descriptors
        else:
            return {"logits": logits,
                    "prob": prob, 
                    "descriptors_raw": descriptors_raw, 
                    "descriptors": descriptors}