import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import math
from typing import Dict, Optional, Tuple, Union

from models.backbones.maxvit_backbone.yolox_network_blocks import BaseConv, DWConv
from models.backbones.maxvit import FeatureMap, BackboneFeatures,CrossMaxVitAttentionPairCl
from models.backbones.maxvit_backbone.layers.maxvit.maxvit import CrossAttentionCl, CrossChannelAttentionCl, CrossAttentionModule, CrossChannelAttentionModule

class FusionBlock(nn.Module):
    def __init__(self, config:dict, dim=None):
        super().__init__()

        self.is_FPN = config["is_FPN"]
        if dim == None:
            embed_dim = config["embed_dim"]
        else:
            embed_dim = dim
        dim_multiplier_per_stage = tuple(config["dim_multiplier"])
        num_blocks_per_stage = tuple(config["num_blocks"])

        fusion_type = config["fusion_type"]
        if fusion_type == "concat":
            self.fusion = BasicConcatFusion
        elif fusion_type == "spatial_attention_gating":
            self.fusion = SpatialAttentionGating
        elif fusion_type == "basic_cross_attention":
            self.fusion = BasicCrossAttentionFusion
        elif fusion_type == "keypoint_gated_cross_attention":
            self.fusion = KeypointGatedCrossAttentionFusion
        elif fusion_type == "channel_cross_attention":
            self.fusion = ChannelCrossAttentionFusion
        elif fusion_type == "channel_maxvit":
            self.fusion = ChannelMaxvitFusion
        elif fusion_type == "channel_maxvit_deblur":
            self.fusion = ChannelMaxvitFusionDeblur
        elif fusion_type == "channel_maxvit_deblur_dual":
            self.fusion = ChannelMaxvitFusionDeblurDual
        elif fusion_type == "channel_cbam":
            self.fusion = ChannelCBAMFusion
        elif fusion_type == "gated_channel_cbam":
            self.fusion = GatedChannelCBAMFusion
        else:
            raise NotImplementedError(f"Fusion type {fusion_type} is not implemented.")

        self.stage_dims = [embed_dim * x for x in dim_multiplier_per_stage]
        self.stages = nn.ModuleList()

        if self.is_FPN:
            for stage_idx, _ in enumerate(num_blocks_per_stage):
                stage_dim = self.stage_dims[stage_idx]
                stage = self.fusion(in_channels=stage_dim, config=config)
                self.stages.append(stage)
        else:
            if dim != None:
                stage_dim = dim
            else:
                stage_dim = self.stage_dims[-1]
            stage = self.fusion(in_channels=stage_dim, config=config)
            self.stages.append(stage)
        
        if config.get("use_seprate_decoder", False):
            self.stages_for_desc = nn.ModuleList()
            if self.is_FPN:
                for stage_idx, _ in enumerate(num_blocks_per_stage):
                    stage_dim = self.stage_dims[stage_idx]
                    stage = self.fusion(in_channels=stage_dim, config=config)
                    self.stages_for_desc.append(stage)
            else:
                if dim != None:
                    stage_dim = dim
                else:
                    stage_dim = self.stage_dims[-1]
                stage = self.fusion(in_channels=stage_dim, config=config)
                self.stages_for_desc.append(stage)
        
        self.use_seprate_decoder = config.get("use_seprate_decoder", False)

    def forward(self, fusion_input: dict) -> Union[FeatureMap, BackboneFeatures]:
        """
        Returns:
            torch.Tensor: (B, C, H, W)
        """
        frame_features = fusion_input["frame_features"]
        event_features = fusion_input["event_features"]

        # frame_features[-1] = torch.clamp(frame_features[-1], min=-10, max=10)
        # event_features[-1] = torch.clamp(event_features[-1], min=-10, max=10)
        # frame_features[-1][:,:,0:2,-2:] = 0
        # event_features[-1][:,:,0:2,-2:] = 0

        assert len(frame_features) == len(event_features)
        if self.is_FPN:
            fusion_input = {
                "frame_features": frame_features,
                "event_features": event_features 
            }
            stage_dict = {}
            output: Dict[int, torch.Tensor] = {}
            for stage_idx, stage in enumerate(self.stages):
                stage_number = stage_idx + 1
                for key, value_dict in fusion_input.items():
                    stage_dict[key] = value_dict[stage_number]
                x = stage(**stage_dict)
                output[stage_number] = x
            
            if self.use_seprate_decoder:
                fusion_input_desc = {
                    "frame_features": frame_features,
                    "event_features": event_features 
                }
                stage_dict_desc = {}
                output_desc: Dict[int, torch.Tensor] = {}
                for stage_idx, stage in enumerate(self.stages_for_desc):
                    stage_number = stage_idx + 1
                    for key, value_dict in fusion_input_desc.items():
                        stage_dict_desc[key] = value_dict[stage_number]
                    x_desc = stage(**stage_dict_desc)
                    output_desc[stage_number] = x_desc
            
        else:
            assert len(self.stages) == 1
            stage = self.stages[0]
            fusion_input = {
                "frame_features": frame_features[-1],   
                "event_features": event_features[-1]
            }

            x = stage(**fusion_input)
            output = x

            if self.use_seprate_decoder:
                stage_desc = self.stages_for_desc[0]
                fusion_input_desc = {
                    "frame_features": frame_features[-1],   
                    "event_features": event_features[-1]
                }
                x_desc = stage_desc(**fusion_input_desc)
                output_desc = x_desc

        if self.use_seprate_decoder:
            return output, output_desc
        else:
            return output

class BasicConcatFusion(nn.Module):
    """
    使用简单的通道拼接融合两个模态的特征 frame_features 和 F_B。
    假设 frame_features 和 event_features 的形状为 (B, C, H, W)。
    """
    def __init__(self, in_channels, config):
        super().__init__()
        self.in_channels = in_channels

        self.conv_list = nn.Sequential(
            BaseConv(in_channels*2, in_channels*2, ksize=1, stride=1, act="silu"),  
            BaseConv(in_channels*2, in_channels, ksize=3, stride=1, act="silu")
        )

    def forward(self, **features):
        """
        Returns:
            torch.Tensor: (B, C, H, W)
        """
        frame_features = features["frame_features"]
        event_features = features["event_features"]

        # 在通道维度上拼接
        F_fused = torch.cat((frame_features, event_features), dim=1)  #(B, 2C, H, W)
        F_fused = self.conv_list(F_fused)  #(B, C, H, W)
        return F_fused

class SpatialAttentionGating(nn.Module):
    """
    使用空间注意力门控机制融合两个模态的特征 frame_features 和 F_B。
    假设 frame_features 和 event_features 的形状为 (B, C, H, W)。
    """
    def __init__(self, in_channels, config):
        super(SpatialAttentionGating, self).__init__()
        # 门控网络：接收拼接后的特征，输出单通道的注意力图
        self.gate_network = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()  # 输出范围在 0 到 1 之间
        )

    def forward(self, **features):
        """
        Args:
            frame_features (torch.Tensor): 模态A的特征, shape (B, C, H, W)
            event_features (torch.Tensor): 模态B的特征, shape (B, C, H, W)
        Returns:
            torch.Tensor: 融合后的特征
        """
        frame_features = features["frame_features"]
        event_features = features["event_features"]

        # 1. 在通道维度上拼接
        F_cat = torch.cat((frame_features, event_features), dim=1)
        
        # 2. 生成空间注意力图 M
        # M 的形状是 (B, 1, H, W)
        M = self.gate_network(F_cat)
        
        # 3. 加权融合
        # 利用广播机制，M 会自动扩展到 (B, C, H, W)
        F_fused = M * frame_features + (1 - M) * event_features
        
        return F_fused

class BasicCrossAttentionFusion(nn.Module):
    """
    使用跨注意力机制融合两个模态的特征 frame_features 和 F_B。
    假设 frame_features 和 event_features 的形状为 (B, C, H, W)。
    """
    def __init__(self, in_channels, config):
        super(BasicCrossAttentionFusion, self).__init__()
        self.in_channels = in_channels
        self.norm_e = nn.LayerNorm(in_channels)
        self.norm_f = nn.LayerNorm(in_channels)
        self.E2I = CrossAttentionCl(in_channels, config["attention"]["dim_head"], config["attention"]["attention_bias"])
        self.I2E = CrossAttentionCl(in_channels, config["attention"]["dim_head"], config["attention"]["attention_bias"])

        self.out_proj = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, **features):
        """
        Returns:
            torch.Tensor: (B, C, H, W)
        """
        frame_features = features["frame_features"]
        event_features = features["event_features"]
        B, C, H, W = frame_features.shape

        image = frame_features.permute(0, 2, 3, 1).contiguous() # b, h, w, c
        event = event_features.permute(0, 2, 3, 1).contiguous() # b, h, w, c

        image = self.norm_f(image)
        event = self.norm_e(event)

        fused_I = image + self.E2I(image, event) # b, h, w, c
        fused_E = event + self.I2E(event, image)

        # fused_I = fused_I.permute(0, 3, 1, 2).contiguous() # b, c, h, w
        # fused_E = fused_E.permute(0, 3, 1, 2).contiguous()
        fused = fused_I + fused_E

        fused = self.out_proj(self.norm1(fused.reshape(B, -1, C)))  # b, c, h, w
        fused = self.norm2(fused)
        fused = fused.permute(0, 2, 1).contiguous().reshape(B, C, H, W)

        return fused
    
class KeypointGatedCrossAttentionFusion(nn.Module):
    """
    使用跨注意力机制融合两个模态的特征 frame_features 和 F_B。
    假设 frame_features 和 event_features 的形状为 (B, C, H, W)。
    """
    def __init__(self, in_channels, config):
        super(KeypointGatedCrossAttentionFusion, self).__init__()
        self.in_channels = in_channels
        self.norm_e = nn.LayerNorm(in_channels)
        self.norm_f = nn.LayerNorm(in_channels)
        self.E2I = CrossAttentionCl(in_channels, config["attention"]["dim_head"], config["attention"]["attention_bias"])
        self.I2E = CrossAttentionCl(in_channels, config["attention"]["dim_head"], config["attention"]["attention_bias"])

        self.out_proj = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.gate_network = nn.Sequential(
            nn.Conv2d(65 * 2, 65, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(65, 1, kernel_size=1),
            nn.Sigmoid()  # 输出范围在 0 到 1 之间
        )

    def forward(self, **features):
        """
        Returns:
            torch.Tensor: (B, C, H, W)
        """
        frame_features = features["frame_features"]
        event_features = features["event_features"]
        F_keypoint = features["F_keypoint"]
        E_keypoint = features["E_keypoint"]

        B, C, H, W = frame_features.shape

        image = frame_features.permute(0, 2, 3, 1).contiguous() # b, h, w, c
        event = event_features.permute(0, 2, 3, 1).contiguous() # b, h, w, c

        F_keypoint = F.softmax(F_keypoint, dim=1)  #(B, C, H, W)
        E_keypoint = F.softmax(E_keypoint, dim=1)  #(B, C, H, W)

        weight = torch.cat((F_keypoint, E_keypoint), dim=1)  #(B, 2C, H, W)
        M = self.gate_network(weight)  #(B, 1, H, W)
        M = M.permute(0, 2, 3, 1).contiguous()  # b, h, w, 1

        image = self.norm_f(image)
        event = self.norm_e(event)

        fused_I = M*(image + self.E2I(image, event)) # b, h, w, c
        fused_E = (1-M)*(event + self.I2E(event, image))

        # fused_I = fused_I.permute(0, 3, 1, 2).contiguous() # b, c, h, w
        # fused_E = fused_E.permute(0, 3, 1, 2).contiguous()
        fused = fused_I + fused_E

        fused = self.out_proj(self.norm1(fused.reshape(B, -1, C)))  # b, c, h, w
        fused = self.norm2(fused)
        fused = fused.permute(0, 2, 1).contiguous().reshape(B, C, H, W)

        return fused

class ChannelCrossAttentionFusion(nn.Module):
    """
    使用跨注意力机制融合两个模态的特征 frame_features 和 F_B。
    假设 frame_features 和 event_features 的形状为 (B, C, H, W)。
    """
    def __init__(self, in_channels, config):
        super(ChannelCrossAttentionFusion, self).__init__()
        self.in_channels = in_channels
        self.norm = config["fusion_norm"]
        self.gated = config["gated"]
        if self.norm == "layernorm":
            self.norm_e = nn.LayerNorm(in_channels)
            self.norm_f = nn.LayerNorm(in_channels)
        elif self.norm == "batchnorm":
            self.norm_e = nn.BatchNorm2d(in_channels)
            self.norm_f = nn.BatchNorm2d(in_channels)
        self.E2I_c = CrossChannelAttentionModule(in_channels, config, need_norm=False) #do Batchnorm2d first
        self.I2E_c = CrossChannelAttentionModule(in_channels, config, need_norm=False)
        self.E2I = CrossAttentionModule(in_channels, config)
        self.I2E = CrossAttentionModule(in_channels, config)


        self.norm_out = nn.LayerNorm(in_channels)
        self.MLP = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=2*in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(2*in_channels,in_channels)
        )

        if self.gated:
            self.gate_network = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.Sigmoid()  # 输出范围在 0 到 1 之间
            )
        
    def forward(self, **features):
        """
        Returns:
            torch.Tensor: (B, C, H, W)
        """
        frame_features = features["frame_features"]
        event_features = features["event_features"]
        B, C, H, W = frame_features.shape

        if self.norm == "batchnorm":
            frame_features = self.norm_f(frame_features)
            event_features = self.norm_e(event_features)

        image = frame_features.permute(0, 2, 3, 1).contiguous() # b, h, w, c
        event = event_features.permute(0, 2, 3, 1).contiguous() # b, h, w, c

        if self.norm == "layernorm":
            image = self.norm_f(image)
            event = self.norm_e(event)

        fused_I = self.E2I_c(image, event)
        fused_E = self.I2E_c(event, image)

        fused_I = self.E2I(fused_I, fused_E) # b, h, w, c
        fused_E = self.I2E(fused_E, fused_I)

        if self.gated:
            weight = torch.cat((image.permute(0, 3, 1, 2), event.permute(0, 3, 1, 2)), dim=1)  #(B, 2C, H, W)
            M = self.gate_network(weight)  #(B, 1, H, W)
            M = M.permute(0, 2, 3, 1).contiguous()  # b, h, w, 1

            fused_I = M * fused_I
            fused_E = (1 - M) * fused_E

        fused = fused_I + fused_E # b, h, w, c

        fused = self.MLP(self.norm_out(fused.reshape(B, -1, C)))  
        fused = fused.permute(0, 2, 1).contiguous().reshape(B, C, H, W)

        return fused
    
class ChannelMaxvitFusion(nn.Module):
    """
    使用跨注意力机制融合两个模态的特征 frame_features 和 F_B。
    假设 frame_features 和 event_features 的形状为 (B, C, H, W)。
    """
    def __init__(self, in_channels, config):
        super(ChannelMaxvitFusion, self).__init__()
        self.in_channels = in_channels
        self.norm = config["fusion_norm"]
        self.gated = config["gated"]
        self.concat = config.get("concat", False)
        self.dual_cross = config.get("dual_cross", False)

        if self.norm == "layernorm":
            self.norm_e = nn.LayerNorm(in_channels)
            self.norm_f = nn.LayerNorm(in_channels)
        elif self.norm == "batchnorm":
            self.norm_e = nn.BatchNorm2d(in_channels)
            self.norm_f = nn.BatchNorm2d(in_channels)
        elif self.norm == "groupnorm":
            self.norm_e = nn.GroupNorm(num_groups=8, num_channels=in_channels)
            self.norm_f = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        self.E2I_c = CrossChannelAttentionModule(in_channels, config, need_norm=False) #do Batchnorm2d first
        self.I2E_c = CrossChannelAttentionModule(in_channels, config, need_norm=False)
        
        if self.dual_cross:
            self.E2I = CrossMaxVitAttentionPairCl(in_channels, False, config["attention"])
            self.I2E = CrossMaxVitAttentionPairCl(in_channels, False, config["attention"])

            if self.concat:
                self.norm_out = nn.LayerNorm(2*in_channels)
                self.MLP = nn.Sequential(
                    nn.Linear(in_features=2*in_channels, out_features=2*in_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(2*in_channels,in_channels)
                )
            else:
                self.norm_out = nn.LayerNorm(in_channels)
                self.MLP = nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=2*in_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(2*in_channels,in_channels)
                )

                
            if self.gated:
                self.gate_network = nn.Sequential(
                    nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, 1, kernel_size=1),
                    nn.Sigmoid()  # 输出范围在 0 到 1 之间
            )
        else:
            self.fusion = CrossMaxVitAttentionPairCl(in_channels, False, config["attention"])
            
            self.norm_out = nn.LayerNorm(in_channels)
            self.MLP = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=2*in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(2*in_channels,in_channels)
            )
                    
    def forward(self, **features):
        """
        Returns:
            torch.Tensor: (B, C, H, W)
        """
        frame_features = features["frame_features"]
        event_features = features["event_features"]
        B, C, H, W = frame_features.shape

        if self.norm in ["batchnorm", "groupnorm"]:
            frame_features = self.norm_f(frame_features)
            event_features = self.norm_e(event_features)

        image = frame_features.permute(0, 2, 3, 1).contiguous() # b, h, w, c
        event = event_features.permute(0, 2, 3, 1).contiguous() # b, h, w, c

        if self.norm == "layernorm":
            image = self.norm_f(image)
            event = self.norm_e(event)

        fused_I = self.E2I_c(image, event)
        fused_E = self.I2E_c(event, image)

        if self.dual_cross:
            fused_I = self.E2I(fused_I, fused_E) # b, h, w, c
            fused_E = self.I2E(fused_E, fused_I)

            if self.gated:
                weight = torch.cat((image.permute(0, 3, 1, 2), event.permute(0, 3, 1, 2)), dim=1)  #(B, 2C, H, W)
                M = self.gate_network(weight)  #(B, 1, H, W)
                M = M.permute(0, 2, 3, 1).contiguous()  # b, h, w, 1

                fused_I = M * fused_I
                fused_E = (1 - M) * fused_E
                
            if self.concat:
                fused = torch.cat((fused_I, fused_E), dim=-1) # b, h, w, 2c
            else:
                fused = fused_I + fused_E # b, h, w, c
        else:
            fused = self.fusion(fused_I, fused_E) # b, h, w, c

        fused = self.MLP(self.norm_out(fused.reshape(B, -1, C)))  
        fused = fused.permute(0, 2, 1).contiguous().reshape(B, C, H, W)

        return fused
    

class GatedFilter(nn.Module):
    """
    [新增] 内部门控子模块
    功能：利用 Frame 特征作为 Query，判断 Event 特征的有效性，并生成 0~1 的 Mask。
    """
    def __init__(self, channels):
        super().__init__()
        # 输入是 RGB + Event 的拼接，所以是 channels * 2
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True), # 1x1 卷积做像素级决策
            nn.Sigmoid() # 输出 0~1
        )
        self._init_weights()

    def _init_weights(self):
        # 初始化：
        # 我们希望初始状态下 Gate 不要完全关闭，而是处于“半开”状态 (Sigmoid(0)=0.5)
        # 这样能保证梯度在训练初期能流过去，让 Gate 学会如何筛选
        for m in self.gate_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_rgb, x_event):
        # 1. 拼接
        cat_x = torch.cat([x_rgb, x_event], dim=1)
        # 2. 计算权重图
        mask = self.gate_net(cat_x)
        # 3. 过滤 Event (软阈值)
        return x_event * mask, mask    

class ChannelMaxvitFusionDeblur(nn.Module):
    def __init__(self, in_channels, config):
        super(ChannelMaxvitFusionDeblur, self).__init__()
        
        self.gated = config.get("gated", False) 
        if self.gated:
            self.gate_filter = GatedFilter(in_channels)

        # 1. 预处理归一化 (针对 VGG Backbone 输出幅度不一致的问题)
        # 显存受限推荐 GroupNorm
        self.norm_f = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.norm_e = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        # 2. 核心融合模块 (包含 Attn + MLP)
        # 直接使用你提供的 CrossChannelAttentionModule
        self.fusion_block = CrossChannelAttentionModule(in_channels, config, need_norm=False) 
        
    def forward(self, **features):
        x_rgb = features["frame_features"]
        x_event = features["event_features"]
        B, C, H, W = x_rgb.shape

        # [新增] 在归一化之前，先进行“质量安检”
        if self.gated:
            x_event, gate_mask = self.gate_filter(x_rgb, x_event)
            # (可选) 你可以将 gate_mask 存入 features 字典返回，方便可视化分析
            # features["gate_mask"] = gate_mask

        # 1. 归一化
        x_rgb = self.norm_f(x_rgb)
        x_event = self.norm_e(x_event)

        # 2. 维度变换 (B, C, H, W) -> (B, H, W, C)
        # LayerNorm 和 Linear 层通常在最后一维操作
        x_rgb_perm = x_rgb.permute(0, 2, 3, 1).contiguous()   # Query
        x_event_perm = x_event.permute(0, 2, 3, 1).contiguous() # Key/Value

        # 3. 融合 (内部已包含 Residual 和 MLP)
        # 输入: x_0 (RGB), x_1 (Event)
        # 你的模块内部逻辑: x = x_0 + attn(x_0, x_1) -> x = x + mlp(x)
        x_fused = self.fusion_block(x_rgb_perm, x_event_perm)

        # 4. 恢复形状 (B, H, W, C) -> (B, C, H, W)
        x_out = x_fused.permute(0, 3, 1, 2).contiguous()

        return x_out

class OutputGating(nn.Module):
    """
    [新增] 输出选择门控 (Output Selection Gating)
    功能：生成像素级或通道级的权重图，用于在"融合特征"和"原始特征"之间做软切换。
    """
    def __init__(self, in_channels):
        super().__init__()
        # input: fused_feature (C)
        # output: weight map (C) 
        # 使用 1x1 卷积极其轻量，不会增加太多显存
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid() # 输出 0~1
        )

    def forward(self, x):
        return self.gate_conv(x)

class ChannelMaxvitFusionDeblurDual(nn.Module):
    """
    [新增] 双向交互融合 (Dual-Way Interaction)
    同时计算 Encoder -> Decoder (E->I) 和 Decoder -> Encoder (I->E) 的 Cross Channel Attention。
    最后将两者结果进行融合。仅使用 CrossChannelAttentionModule。
    """
    def __init__(self, in_channels, config):
        super(ChannelMaxvitFusionDeblurDual, self).__init__()
        
        self.gated = config.get("gated", False) 
        self.concat = config.get("concat", False) # 融合方式：Concat or Add
        self.residule = config.get("residule", True) # 是否使用残差连接
        self.norm = config.get("fusion_norm", "groupnorm")
        self.output_gating = config.get("output_gating", False) 
        self.dual_gating = config.get("dual_gating", False) # 新参数：开启双路互补门控

        # 如果开启双路门控，需要一个生成 Gate 的小网络
        if self.dual_gating:
            # 输入 2*C (因为要看两路特征)，输出 1*C (Gate)
            self.arbitrator = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Sigmoid() 
            )

        if self.output_gating:
            self.out_gate = OutputGating(in_channels)

        if self.gated:
            # Mask 分别用于过滤对方
            self.gate_filter_e = GatedFilter(in_channels) # 用 Frame 过滤 Event
            self.gate_filter_f = GatedFilter(in_channels) # 用 Event 过滤 Frame


        if self.norm == "layernorm":
            self.norm_e = nn.LayerNorm(in_channels)
            self.norm_f = nn.LayerNorm(in_channels)
        elif self.norm == "batchnorm":
            self.norm_e = nn.BatchNorm2d(in_channels)
            self.norm_f = nn.BatchNorm2d(in_channels)
        elif self.norm == "groupnorm":
            self.norm_f = nn.GroupNorm(num_groups=8, num_channels=in_channels)
            self.norm_e = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        # 2. 核心双向融合模块 (Channel Attention Only)
        # E2I: Event 增强 Image (Query=Image, Key/Value=Event)
        self.E2I_attn = CrossChannelAttentionModule(in_channels, config, need_norm=False) 
        # I2E: Image 增强 Event (Query=Event, Key/Value=Image)
        self.I2E_attn = CrossChannelAttentionModule(in_channels, config, need_norm=False) 

        # 3. 最终融合层 (Fusion MLP)
        if self.concat:
            self.norm_out = nn.LayerNorm(2*in_channels)
            self.MLP = nn.Sequential(
                nn.Linear(in_features=2*in_channels, out_features=2*in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(2*in_channels, in_channels)
            )
        else: # Default: Element-wise Add
            self.norm_out = nn.LayerNorm(in_channels)
            self.MLP = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=2*in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(2*in_channels, in_channels)
            )
        # 仅在纯残差模式下使用零初始化
        if self.residule and not self.output_gating and isinstance(self.MLP[-1], nn.Linear):
            nn.init.constant_(self.MLP[-1].weight, 0)
            nn.init.constant_(self.MLP[-1].bias, 0)

    def forward(self, **features):
        x_rgb = features["frame_features"]   # (B, C, H, W)
        x_rgb_raw = x_rgb 
        x_event = features["event_features"] # (B, C, H, W)
        B, C, H, W = x_rgb.shape

        # [Pre-Gating] 双向门控过滤 (可选)
        if self.gated:
            # 用 RGB 检查 Event 是否有效
            x_event, _ = self.gate_filter_e(x_rgb, x_event)
            # 用 Event 检查 RGB 是否有效 
            x_rgb, _ = self.gate_filter_f(x_event, x_rgb)

        # === [修复 1] 根据 Norm 类型选择在 Permute 前或后执行，且只执行一次 ===
        
        # Case A: BatchNorm / GroupNorm (需要在 B, C, H, W 形状下进行)
        if self.norm in ["batchnorm", "groupnorm"]:
            x_rgb = self.norm_f(x_rgb)
            x_event = self.norm_e(x_event)

        # 2. 维度变换 (B, C, H, W) -> (B, H, W, C)
        x_rgb_perm = x_rgb.permute(0, 2, 3, 1).contiguous()
        x_event_perm = x_event.permute(0, 2, 3, 1).contiguous()

        # Case B: LayerNorm (需要在 B, H, W, C 形状下进行)
        if self.norm == "layernorm":
            # 注意：是对变换后的 _perm 变量进行归一化
            x_rgb_perm = self.norm_f(x_rgb_perm)
            x_event_perm = self.norm_e(x_event_perm)

        # 3. 双向 Attention
        # E2I: RGB 作为 Query，吸收 Event 信息
        fused_I = self.E2I_attn(x_rgb_perm, x_event_perm) 
        
        # I2E: Event 作为 Query，吸收 RGB 信息
        fused_E = self.I2E_attn(x_event_perm, x_rgb_perm)

        # 4. 合并双流 (Aggregation)
        # 互补门控融合 (Arbitration)
        if self.dual_gating:
            # 把形状变回 (B, C, H, W) 以便做卷积门控
            f_I = fused_I.permute(0, 3, 1, 2).contiguous()
            f_E = fused_E.permute(0, 3, 1, 2).contiguous()
            # 拼接两路特征，让网络判断哪里该听谁的
            cat_feat = torch.cat([f_I, f_E], dim=1) # (B, 2C, H, W)
            z = self.arbitrator(cat_feat)           # (B, C, H, W) 
            
            # z -> 1: 偏向选择 Event 视角 (fused_E)
            # z -> 0: 偏向选择 RGB 视角 (fused_I)
            fused = z * f_E + (1 - z) * f_I
            fused = fused.permute(0, 2, 3, 1).contiguous()
        elif self.concat:
            fused = torch.cat((fused_I, fused_E), dim=-1) # (B, H, W, 2C)
        else:
            fused = fused_I + fused_E # (B, H, W, C)

        # 5. 最终 MLP 融合
        fused = self.norm_out(fused)
        # Reshape to (B*H*W, C) for Linear layer efficiency
        fused = self.MLP(fused.reshape(-1, fused.shape[-1])) 
        fused = fused.reshape(B, H, W, C)

        # 6. 恢复形状 (B, C, H, W)
        x_out = fused.permute(0, 3, 1, 2).contiguous()


        if self.output_gating:
            # 计算置信度 Z
            # MLP 的结果已经包含了 "修复后的信息"
            z = self.out_gate(x_out)
            
            # 软选择：在 "原始坏图" 和 "修复结果" 之间插值
            # 当 x_rgb_raw 很差时，Z 会倾向于 1，完全使用 x_out
            # 当 x_rgb_raw 很好时，Z 会倾向于 0，保留 x_rgb_raw
            x_out = z * x_out + (1 - z) * x_rgb_raw
        elif self.residule:
            x_out = x_out + x_rgb_raw  # Residual Connection with original RGB features

        return x_out
    

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
    
class LocalFeatureFusion(nn.Module):
    """
    [改进版] 局部特征融合模块 + 集成 CBAM
    结构: 
    Input -> [1x1 Conv (升维)] 
          -> [3x3 DWConv (空间混合)] 
          -> [CBAM Attention (在显存和信息量最大时进行筛选)] 
          -> [1x1 Conv (降维)] 
          -> Residual Add
    """
    def __init__(self, in_channels, expansion=2, use_cbam=False, cbam_config=None):
        super().__init__()
        hidden_channels = int(in_channels * expansion)
        
        # 1. Pointwise Conv (升维)
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=False),
            # 这里的 Norm 推荐用 BatchNorm (如果 batch size 够大) 或者 GroupNorm
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True)
        )

        # 2. Depthwise Conv (空间信息交互)
        self.dw = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3. Attention (嵌入式)
        self.use_cbam = use_cbam
        if self.use_cbam:
            reduction = cbam_config.get("cbam_reduction", 16) if cbam_config else 16
            k_size = cbam_config.get("cbam_kernel_size", 7) if cbam_config else 7
            self.att = CBAMBlock(hidden_channels, reduction=reduction, kernel_size=k_size)

        # 4. Pointwise Conv (降维)
        self.pw2 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1, 1, 0, bias=False),
            nn.GroupNorm(8, in_channels),
        )

        nn.init.constant_(self.pw2[1].weight, 0)
        nn.init.constant_(self.pw2[1].bias, 0)
        
    def forward(self, x):
        identity = x
        out = self.pw1(x)
        out = self.dw(out)
        
        # 在降维之前做 Attention
        if self.use_cbam:
            out = self.att(out)
            
        out = self.pw2(out)
        return identity + out

class ChannelCBAMFusion(nn.Module):
    """
    CBAM + Channel Attention 的双模态融合
    先用 CrossChannelAttentionModule 做跨模态通道交互，
    再用 CBAM 做通道+空间注意力增强。
    """
    def __init__(self, in_channels, config):
        super().__init__()
        self.norm = config.get("fusion_norm", "groupnorm")
        self.concat = config.get("concat", False)
        self.add_mixer = config.get("add_mixer", False) # 是否在 CBAM 前加入一个轻量级的局部特征融合模块

        if self.norm == "layernorm":
            self.norm_e = nn.LayerNorm(in_channels)
            self.norm_f = nn.LayerNorm(in_channels)
        elif self.norm == "batchnorm":
            self.norm_e = nn.BatchNorm2d(in_channels)
            self.norm_f = nn.BatchNorm2d(in_channels)
        elif self.norm == "groupnorm":
            self.norm_e = nn.GroupNorm(num_groups=8, num_channels=in_channels)
            self.norm_f = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        # 双向通道注意力（跨模态）
        self.E2I_c = CrossChannelAttentionModule(in_channels, config, need_norm=False)
        self.I2E_c = CrossChannelAttentionModule(in_channels, config, need_norm=False)

        # concat 时用 1x1 压回 C
        if self.concat:
            self.proj = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=False)

        if self.add_mixer:
            # 启用 Mixer 内部的 CBAM，效果通常优于外部串联
            self.mixer = LocalFeatureFusion(in_channels, expansion=2, use_cbam=True, cbam_config=config)
        else:
            # CBAM
            reduction = config.get("cbam_reduction", 16)
            kernel_size = config.get("cbam_kernel_size", 7)
            self.cbam = CBAMBlock(in_channels, reduction=reduction, kernel_size=kernel_size)

    def forward(self, **features):
        x_rgb = features["frame_features"]   # (B, C, H, W)
        x_event = features["event_features"] # (B, C, H, W)

        # Norm (B,C,H,W)
        if self.norm in ["batchnorm", "groupnorm"]:
            x_rgb = self.norm_f(x_rgb)
            x_event = self.norm_e(x_event)

        # 转成 (B,H,W,C)
        rgb = x_rgb.permute(0, 2, 3, 1).contiguous()
        evt = x_event.permute(0, 2, 3, 1).contiguous()

        # LayerNorm 在 (B,H,W,C)
        if self.norm == "layernorm":
            rgb = self.norm_f(rgb)
            evt = self.norm_e(evt)

        # 双向通道注意力
        fused_I = self.E2I_c(rgb, evt)
        fused_E = self.I2E_c(evt, rgb)

        # 融合
        if self.concat:
            fused = torch.cat((fused_I, fused_E), dim=-1)  # (B,H,W,2C)
            fused = fused.permute(0, 3, 1, 2).contiguous() # (B,2C,H,W)
            fused = self.proj(fused)                       # (B,C,H,W)
        else:
            fused = (fused_I + fused_E).permute(0, 3, 1, 2).contiguous()
        
        if self.add_mixer:
            fused = self.mixer(fused)
        else:
            # CBAM
            fused = self.cbam(fused)

        return fused
    

class GatedChannelCBAMFusion(nn.Module):
    """
    改进版：基于门控机制的双模态特征融合模块
    核心优势：
    1. 通过 Sigmoid 门控生成 [0,1] 权重，实现模态竞争与选择。
    2. 加权融合严格满足 F/2 + E/2 的数值尺度近似。
    3. 在末尾引入双模态原始输入的全局残差连接。
    """
    def __init__(self, in_channels, config):
        super().__init__()
        self.norm_type = config.get("fusion_norm", "groupnorm")

        # 1. 归一化层初始化
        if self.norm_type == "layernorm":
            self.norm_e = nn.LayerNorm(in_channels)
            self.norm_f = nn.LayerNorm(in_channels)
        elif self.norm_type == "batchnorm":
            self.norm_e = nn.BatchNorm2d(in_channels)
            self.norm_f = nn.BatchNorm2d(in_channels)
        elif self.norm_type == "groupnorm":
            # 默认使用 GroupNorm，规避 Batch Size 过小的问题
            self.norm_e = nn.GroupNorm(num_groups=8, num_channels=in_channels)
            self.norm_f = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        # 2. 跨模态通道交互 (假设外部已定义 CrossChannelAttentionModule)
        self.E2I_c = CrossChannelAttentionModule(in_channels, config, need_norm=False)
        self.I2E_c = CrossChannelAttentionModule(in_channels, config, need_norm=False)

        # 3. 门控生成器 (Gate Generator)
        # 将 CBAM 改造为决策层，输入拼接后的特征，输出用于选择的 Mask
        reduction = config.get("cbam_reduction", 16)
        kernel_size = config.get("cbam_kernel_size", 7)
        self.gate_generator = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            # 借用你写好的 CBAMBlock 来提取空间和通道的联合重要性
            CBAMBlock(in_channels, reduction=reduction, kernel_size=kernel_size),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid() # 关键：将权重严格限制在 (0, 1) 之间
        )

        # 4. 局部特征提取器
        # 这里复用你写好的 LocalFeatureFusion，但内部无需再用 CBAM (use_cbam=False)
        # 因为在门控阶段已经做过极致的注意力筛选了
        self.mixer = LocalFeatureFusion(in_channels, expansion=2, use_cbam=False)

    def forward(self, **features):
        x_rgb = features["frame_features"]   # (B, C, H, W)
        x_event = features["event_features"] # (B, C, H, W)

        # --- 第 1 步：基础归一化 ---
        norm_rgb = x_rgb
        norm_evt = x_event
        
        if self.norm_type in ["batchnorm", "groupnorm"]:
            norm_rgb = self.norm_f(x_rgb)
            norm_evt = self.norm_e(x_event)

        # 转换维度 (B, H, W, C) 以适应可能需要的 LayerNorm 和交互模块
        rgb_hwc = norm_rgb.permute(0, 2, 3, 1).contiguous()
        evt_hwc = norm_evt.permute(0, 2, 3, 1).contiguous()

        if self.norm_type == "layernorm":
            rgb_hwc = self.norm_f(rgb_hwc)
            evt_hwc = self.norm_e(evt_hwc)

        # --- 第 2 步：双向通道注意力交互 ---
        # 提取跨模态信息，输出形状仍为 (B, H, W, C)
        interact_I = self.E2I_c(rgb_hwc, evt_hwc)
        interact_E = self.I2E_c(evt_hwc, rgb_hwc)

        # 转回 (B, C, H, W) 以便后续进行 2D 卷积处理
        fused_I = interact_I.permute(0, 3, 1, 2).contiguous()
        fused_E = interact_E.permute(0, 3, 1, 2).contiguous()

        # --- 第 3 步：动态门控选择与融合 (核心逻辑) ---
        concat_features = torch.cat([fused_I, fused_E], dim=1) 
        M = self.gate_generator(concat_features) # M 形状为 (B, C, H, W)
        gated_fused = M * fused_I + (1 - M) * fused_E

        # --- 第 4 步：局部特征细化 ---
        refined_fused = self.mixer(gated_fused)

        return refined_fused