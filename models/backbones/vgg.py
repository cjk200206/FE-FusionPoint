# SuperPoint BN blocks adapted from rpautrat/SuperPoint (MIT).
# Copyright (c) 2018 Paul-Edouard Sarlin & Rémi Pautrat.
# Full upstream MIT notice: models/superpoint.py.

import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from torchvision.ops import DeformConv2d


class VggBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, activate=True, norm = "bn"):
        super().__init__()
        self.activate = activate
        if norm == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm == "gn":
            norm_layer = lambda num_channels: nn.GroupNorm(num_groups=8, num_channels=num_channels)
        elif norm == "in":
            norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError(f"Unsupported normalization type: {norm}")

        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            norm_layer(output_channels)
            )
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layers(x)
        if self.activate:
            x = self.activation(x)
        return x

class VggTransposeBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, activate=True):
        super().__init__()
        self.activate = activate
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=1, padding=1),  # padding of 1 preserves size
            nn.BatchNorm2d(output_channels)
            )
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layers(x)
        if self.activate:
            x = self.activation(x)
        return x

class VggBackbone(nn.Module):
    def __init__(self, input_channels, output_channels, return_maxpool_indeces=False, norm="bn", enable_dog=False, enable_selector=False):
        super().__init__()
        self.return_maxpool_indeces = return_maxpool_indeces
        self.enable_dog = enable_dog # [修改] 保存开关状态
        self.enable_selector = enable_selector # [新增] 时间尺度选择器开关

        if self.enable_selector:
            print(f"VggBackbone: Time Scale Selector Enabled for {input_channels} scales.")
            self.scale_selector = TimeScaleSelector(input_channels)

        # [新增] 根据开关初始化 DoG 模块
        # 第一层 VggBlock 输出通道固定为 64，所以这里是 64
        if self.enable_dog:
            self.dog_attention = MultiScaleDoGAttention(64)

        self.layers = nn.ModuleList([
            VggBlock(input_channels, 64, 3, norm=norm),
            VggBlock(64, 64, 3, norm=norm), #1
            nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=return_maxpool_indeces),

            VggBlock(64, 64, 3, norm=norm),
            VggBlock(64, 64, 3, norm=norm), #4
            nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=return_maxpool_indeces),

            VggBlock(64, 128, 3, norm=norm),
            VggBlock(128, 128, 3, norm=norm), #7
            nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=return_maxpool_indeces),

            VggBlock(128, 128, 3, norm=norm),
            VggBlock(128, output_channels, 3, norm=norm) #10
        ])

    def forward(self, x):
        # [新增] 0. 动态选择最佳时间尺度
        if self.enable_selector:
            x = self.scale_selector(x)
            
        outputs = []
        maxpool_indeces = []
        for layer_idx, layer in enumerate(self.layers):
            if self.return_maxpool_indeces and type(layer).__name__ == "MaxPool2d":
                out, ind = layer(outputs[-1])
                outputs.append(out)
                maxpool_indeces.append(ind)
            else:
                x = layer(x)
                
                # [新增] 如果启用了 DoG，在第一个 Block (layer_idx=0) 后立即执行
                if self.enable_dog and layer_idx == 0:
                    x = self.dog_attention(x)

                if layer_idx in [1,4,7,10]:
                    outputs.append(x)
        return outputs, maxpool_indeces
    
class VggBackbone_Upsample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MaxUnpool2d(kernel_size = 2, stride = 2),
            VggTransposeBlock(input_channels + 128, 128, 3),
            VggTransposeBlock(128, 64, 3),

            nn.MaxUnpool2d(kernel_size = 2, stride = 2),
            VggTransposeBlock(128, 64, 3),
            VggTransposeBlock(64, 64, 3),

            nn.MaxUnpool2d(kernel_size = 2, stride = 2),
            VggTransposeBlock(128, 128, 3),
            VggTransposeBlock(128, output_channels, 3)
        ])

    def forward(self, outputs, maxpool_indeces):
        maxpool_indeces_reversed_it = reversed(maxpool_indeces)
        down_outputs_reversed_it = reversed(outputs)
        skip_connection_in_next_layer = False

        # Skip results of last two layers
        next(down_outputs_reversed_it)
        next(down_outputs_reversed_it)

        x = outputs[-1]
        for layer in self.layers:
            output_down = next(down_outputs_reversed_it)
            if type(layer).__name__ == "MaxUnpool2d":
                x = layer(x, next(maxpool_indeces_reversed_it))
                skip_connection_in_next_layer = True
            else:
                if skip_connection_in_next_layer:
                    x = torch.concat((x, output_down), dim=1)  # skip connection
                    skip_connection_in_next_layer = False
                x = layer(x)
        return x
    

class SuperPointBackboneBatchNormed(nn.Module):
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }

    def __init__(self, config):
        super().__init__()

        self.config = {**self.default_config, **config}
        self.stride = 8
        self.channels = [64, 64, 128, 128, 256]
        self.descriptor_dim = self.config["descriptor_dim"]

        self.backbone = nn.Sequential(
            # Block 0: 输入 -> 64 (Key: backbone.0.*)
            nn.Sequential(
                VGGBN(1, 64, 3),
                VGGBN(64, 64, 3),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),


            # Block 1: 64 -> 64 (Key: backbone.1.*)
            nn.Sequential(
                VGGBN(64, 64, 3),
                VGGBN(64, 64, 3),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),

            # Block 2: 64 -> 128 (Key: backbone.2.*)
            nn.Sequential(
                VGGBN(64, 128, 3),
                VGGBN(128, 128, 3),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),

            # Block 3: 128 -> 128 (Key: backbone.3.*)
            # 注意：最后一个 Block 没有 MaxPool，这正好符合你的 logic
            nn.Sequential(
                VGGBN(128, 128, 3),
                VGGBN(128, 128, 3),
            ),
        )

        c = self.channels[-1]
        self.detector = nn.Sequential(
            VGGBN(128, c, 3),
            VGGBN(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBN(128, c, 3),
            VGGBN(c, self.descriptor_dim, 1, relu=False),
        )

    def forward(self, image):

        x = image
        intermediate_features = []
        
        for block_idx, block in enumerate(self.backbone):
            
            if isinstance(block[-1], nn.MaxPool2d):
                for layer in block[:-1]:
                    x = layer(x)
                
                intermediate_features.append(x)
                
                x = block[-1](x)
            else:
                x = block(x)

        intermediate_features.append(x)
        
        features = x

        desc = self.descriptor(features)
        desc_expand = torch.nn.functional.interpolate(
            desc, scale_factor=self.stride, mode="bilinear"
        )
        desc_expand = torch.nn.functional.normalize(desc_expand, dim=1)

        desc = torch.nn.functional.normalize(desc, dim=1)

        # Decode the detection scores
        semi = self.detector(features)
        scores = torch.nn.functional.softmax(semi, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * self.stride, w * self.stride
        )
        

        output = {"logits": semi,
                "descriptors_raw": desc,
                "descriptors": desc_expand,
                "features": intermediate_features,
                "prob": scores}

        return output


class VGGBN(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding
        )
        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        bn = nn.BatchNorm2d(c_out, eps=0.001)
        super().__init__(
            OrderedDict(
                [
                    ("conv", conv),
                    ("activation", activation),
                    ("bn", bn),
                ]
            )
        )

class SPBNBackboneWithEvent(nn.Module):
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "channels": [64, 64, 128, 128, 256],
        # [New] 新增配置：Event 特征的通道列表，例如 [64, 128]
        # 如果不提供或为空，则不创建注入层
        "event_channels_list": [64, 64, 128], 
        "injection_idxs": [0, 1, 2],  # 在哪些 Block 之后进行注入
    }

    def __init__(self, config):
        super().__init__()

        self.config = {**self.default_config, **config}
        self.stride = 8
        self.channels = [64, 64, 128, 128, 256]
        self.descriptor_dim = self.config["descriptor_dim"]

        # ================= Original Backbone Definition =================
        self.backbone = nn.Sequential(
            # Block 0: 1 -> 64
            nn.Sequential(
                VGGBN(1, 64, 3),
                VGGBN(64, 64, 3),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            # Block 1: 64 -> 64
            nn.Sequential(
                VGGBN(64, 64, 3),
                VGGBN(64, 64, 3),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            # Block 2: 64 -> 128
            nn.Sequential(
                VGGBN(64, 128, 3),
                VGGBN(128, 128, 3),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            # Block 3: 128 -> 128
            nn.Sequential(
                VGGBN(128, 128, 3),
                VGGBN(128, 128, 3),
            ),
        )

        self.detector = nn.Sequential(
            VGGBN(128, 256, 3),
            VGGBN(256, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBN(128, 256, 3),
            VGGBN(256, self.descriptor_dim, 1, relu=False),
        )

        # ================= [New] Injection Layers =================
        self.adapters = nn.ModuleList()
        event_chs = self.config.get("event_channels_list", [])
        injection_idxs = self.config.get("injection_idxs", [])
        
        # 定义 RGB Backbone 每个 Block 输出的通道数 (用于对齐)
        # Block 0 -> 64
        # Block 1 -> 64
        # Block 2 -> 128
        # Block 3 -> 128
        rgb_block_channels = self.config["channels"][:-1]  # [64, 64, 128, 128]

        injection_type = config.get("injection_type", "conv")
        if injection_type == "conv":
            self.injection_module = InjectionLayer
        elif injection_type == "deformable":
            self.injection_module = DeformableInjection  # 你也可以选择 InjectionLayer
        elif injection_type == "deformable_e2f":
            self.injection_module = DeformableE2FInjection
        elif injection_type == "selective_e2f":
            self.injection_module = SelectiveE2FInjection
        elif injection_type == "selective_e2f_new":
            self.injection_module = SelectiveE2FInjection_New

        if len(event_chs) > 0:
            for i, e_dim in enumerate(event_chs):
                if i < len(rgb_block_channels):
                    # 创建注入层：Event通道 -> RGB通道
                    rgb_dim = rgb_block_channels[injection_idxs[i]]
                    if self.injection_module in [DeformableInjection, DeformableE2FInjection, SelectiveE2FInjection]:
                        self.adapters.append(self.injection_module(self.config, e_dim, rgb_dim, rgb_dim//2))
                    elif self.injection_module in [InjectionLayer, SelectiveE2FInjection_New]:
                        self.adapters.append(self.injection_module(self.config, e_dim, rgb_dim))
                else:
                    break
        # ==========================================================

    def forward(self, image, event_features_list=None, return_vis=False):
        """
        Args:
            image: [B, 1, H, W]
            event_features_list: Optional list of tensors from Event Backbone.
                                 e.g., [feat_block0, feat_block1, ...]
        """
        x = image
        intermediate_features = []
        
        # 遍历 Backbone 的每一个 Block (一共4个)
        for block_idx, block in enumerate(self.backbone):
            
            # 1. 运行当前 Block 的卷积层 (Pre-Pooling)
            if isinstance(block[-1], nn.MaxPool2d):
                for layer in block[:-1]: # 跑除了 Pool 之外的层
                    x = layer(x)
                
                # --- [New] 在 Pooling 之前注入 Event 特征 ---
                # 逻辑：我们在特征图分辨率最高的时候注入，信息损失最小
                if event_features_list is not None and block_idx < len(self.adapters):
                    # 取出对应的 Event 特征 (假设 list 顺序和 block 对应)
                    # 如果 event list 比 backbone 短，就只注入前几层
                    if block_idx < len(event_features_list):
                        event_feat = event_features_list[block_idx]
                        adapter = self.adapters[block_idx]
                        x = adapter(x, event_feat) # 注入！
                # -------------------------------------------

                # 收集用于 Loss 的中间特征 (注入后的)
                intermediate_features.append(x)
                
                # 最后执行 Pooling
                x = block[-1](x)
            
            else:
                # 最后一个 Block (没有 Pooling)
                x = block(x)
                
                # --- [New] 最后一个 Block 也要尝试注入 ---
                if event_features_list is not None and block_idx < len(self.adapters):
                     if block_idx < len(event_features_list):
                        event_feat = event_features_list[block_idx]
                        adapter = self.adapters[block_idx]
                        x = adapter(x, event_feat)
                # ---------------------------------------

        # 原代码逻辑：把最后的结果也加进 list
        intermediate_features.append(x)
        
        features = x # [B, 128, H/8, W/8]

        if return_vis:
            # 2. Descriptor Head
            desc = self.descriptor(features)
            desc_expand = torch.nn.functional.interpolate(
                desc, scale_factor=self.stride, mode="bilinear"
            )
            desc_expand = torch.nn.functional.normalize(desc_expand, dim=1)
            desc = torch.nn.functional.normalize(desc, dim=1)

            # 3. Detector Head
            semi = self.detector(features)
            scores = torch.nn.functional.softmax(semi, 1)[:, :-1]
            b, _, h, w = scores.shape
            scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
            scores = scores.permute(0, 1, 3, 2, 4).reshape(
                b, h * self.stride, w * self.stride
            )
            
            output = {
                "logits": semi,
                "descriptors_raw": desc,
                "descriptors": desc_expand,
                "features": intermediate_features, # 这里返回的是包含了 Event 信息的混合特征
                "prob": scores
            }

            return output
        else:
            output = {
                # "logits": semi,
                # "descriptors_raw": desc,
                # "descriptors": desc_expand,
                "features": intermediate_features, # 这里返回的是包含了 Event 信息的混合特征
                # "prob": scores
            }

            return output

class InjectionLayer(nn.Module):
    """
    轻量级注入层：将 Event 特征对齐到 RGB 特征，并相加。
    关键特性：零初始化 (Zero-Init)，保证训练初期完全等同于原始网络。
    """
    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        # 使用 1x1 卷积调整通道数 (Event -> RGB)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 【关键】初始化为 0，确保初始状态下不改变 RGB 特征分布
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x_rgb, x_event):
        """
        x_rgb:   [B, C_rgb, H, W]
        x_event: [B, C_event, H', W']
        """
        # 1. 对 Event 特征进行变换 (C_event -> C_rgb)
        feat = self.conv(x_event)
        
        # 2. 空间尺寸对齐 (以 RGB 为准)
        if feat.shape[-2:] != x_rgb.shape[-2:]:
            feat = F.interpolate(feat, size=x_rgb.shape[-2:], mode='bilinear', align_corners=False)
            
        # 3. 残差注入
        return x_rgb + feat 


class DeformableInjection(nn.Module):
    def __init__(self, config, event_channels, rgb_channels, inner_dim=64):
        super().__init__()
        
        # --- 1. 特征预处理 (降维 + Event去噪) ---
        self.rgb_reduce = nn.Conv2d(rgb_channels, inner_dim, kernel_size=1)
        
        # Event 依然保留 Gated Attention 进行去噪，防止噪声误导 RGB 形变
        self.event_fusion = nn.Sequential(
            nn.Conv2d(event_channels, inner_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(inplace=True)
        )
        self.event_gate = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(inner_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # --- 2. 交叉 Offset 预测器 (核心) ---
        
        # A. 用 Event 指导 RGB 形变 (Feature Deblurring)
        # 输入: Event 特征 -> 输出: RGB DCN 的 offsets
        self.offset_learner_for_rgb = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, padding=1), # 这里只看 Event，或者也可以 concat RGB
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim, 2 * 3 * 3, kernel_size=1) 
        )
        self.dcn_rgb = DeformConv2d(inner_dim, inner_dim, 3, padding=1)

        # B. 用 RGB 指导 Event 形变 (Alignment)
        # 输入: RGB 特征 -> 输出: Event DCN 的 offsets
        self.offset_learner_for_event = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim, 2 * 3 * 3, kernel_size=1) 
        )
        self.dcn_event = DeformConv2d(inner_dim, inner_dim, 3, padding=1)
        
        # --- 4. 融合与注入 ---
        self.fusion = nn.Conv2d(inner_dim * 2, inner_dim, kernel_size=3, padding=1)
        
        # 最后的投影层 (Zero Init)
        self.output_proj = nn.Conv2d(inner_dim, rgb_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        # 1. Offsets 初始化为 0 -> 初始不做形变，退化为普通卷积
        nn.init.constant_(self.offset_learner_for_rgb[-1].weight, 0)
        nn.init.constant_(self.offset_learner_for_rgb[-1].bias, 0)
        nn.init.constant_(self.offset_learner_for_event[-1].weight, 0)
        nn.init.constant_(self.offset_learner_for_event[-1].bias, 0)
        
        # 2. 注入层 Zero Init -> 初始不影响主干
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, x_rgb, x_events):
        # x_rgb: [B, C, H, W]
        # x_events: List of [B, C, H, W]
        
        # 1. 预处理
        feat_rgb = self.rgb_reduce(x_rgb)
        
        feat_event = self.event_fusion(x_events)
        gate = self.event_gate(feat_event)
        feat_event_clean = feat_event * gate # 去噪后的 Event
        
        # 2. 交叉形变 (Cross Deformation)
        
        # --- Path A: Event 指导 RGB (试图修复 RGB 的模糊) ---
        # 逻辑：Event 知道运动轨迹，所以由 feat_event_clean 预测 offsets
        offsets_rgb = self.offset_learner_for_rgb(feat_event_clean)
        feat_rgb_deformed = self.dcn_rgb(feat_rgb, offsets_rgb)
        
        # --- Path B: RGB 指导 Event (试图对齐 Event 到 RGB 结构) ---
        # 逻辑：RGB 知道语义边缘在哪里，由 feat_rgb 预测 offsets
        offsets_event = self.offset_learner_for_event(feat_rgb)
        feat_event_deformed = self.dcn_event(feat_event_clean, offsets_event)
        
        # 3. 融合增强
        # 将修整后的 RGB 和 对齐后的 Event 结合
        # 注意：这里我们不仅注入 Event，还注入了 "经过 Event 修正的 RGB 特征"
        combined = torch.cat([feat_rgb_deformed, feat_event_deformed], dim=1)
        feat_fused = self.fusion(combined)
        
        # 4. 残差注入
        delta = self.output_proj(feat_fused)
        
        return x_rgb + delta

class DeformableE2FInjection(nn.Module):
    def __init__(self, config, event_channels, rgb_channels, inner_dim=64):
        super().__init__()
        
        # --- 1. 特征预处理 (降维 + Event去噪) ---
        self.event_reduce = nn.Sequential(
            nn.Conv2d(event_channels, inner_dim, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=inner_dim),
            nn.ReLU(inplace=True)
        )

        self.event_gate = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(inner_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.event_out_conv = nn.Conv2d(event_channels, event_channels,kernel_size=1)

        # --- 2. Event Offset 预测器 (核心) ---
        
        # 用 Event 指导 RGB 形变 (Feature Deblurring)
        # 输入: Event 特征 -> 输出: RGB DCN 的 offsets
        self.offset_learner_for_rgb = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(inner_dim, 3*3*2, kernel_size=1, stride=1, padding=0),
        )

        self.dcn_rgb = DeformConv2d(rgb_channels, rgb_channels, 3, padding=1)
        self.dcn_rgb_norm = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=rgb_channels),
            nn.ReLU(inplace=True)
        )
        
        # --- 3. RGB Gate 比对去噪前后 ---
        self.rgb_gate = nn.Sequential(
            nn.Conv2d(rgb_channels*2, rgb_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(rgb_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        # 1. Offsets 初始化为 0 (DCN 退化为 Conv3x3)
        nn.init.constant_(self.offset_learner_for_rgb[-1].weight, 0)
        nn.init.constant_(self.offset_learner_for_rgb[-1].bias, 0)
        
        # 2. RGB Gate 初始化为 0 (强制选择原始 RGB)
        nn.init.normal_(self.rgb_gate[-2].weight, std=0.001)
        nn.init.constant_(self.rgb_gate[-2].bias, -5.0) 

        # 3. [关键] Event out Conv 初始化为 0
        # 确保初始时：x_rgb + 0
        nn.init.constant_(self.event_out_conv.weight, 0)
        nn.init.constant_(self.event_out_conv.bias, 0)

    def forward(self, x_rgb, x_events):
        # x_rgb: [B, C, H, W]
        # x_events: List of [B, C, H, W]
        
        # 1. 预处理
        feat_event = self.event_reduce(x_events)

        # 2. 
        # --- Event 指导 RGB (试图修复 RGB 的模糊) ---
        # 逻辑：Event 知道运动轨迹，所以由 feat_event_clean 预测 offsets
        offsets_rgb = self.offset_learner_for_rgb(feat_event)
        x_rgb_deformed = self.dcn_rgb(x_rgb, offsets_rgb)
        x_rgb_deformed = self.dcn_rgb_norm(x_rgb_deformed)

        # 3. rgb Gate 比对去噪前后
        rgb_concat = torch.cat([x_rgb, x_rgb_deformed], dim=1)
        rgb_gate = self.rgb_gate(rgb_concat)
        x_rgb_final = rgb_gate * x_rgb_deformed + (1 - rgb_gate) * x_rgb

        # 4. event注入
        event_gate = self.event_gate(feat_event)
        x_event_final = x_events * event_gate # 去噪后的 Event
        
        return x_rgb_final + self.event_out_conv(x_event_final)
    

class SEBlock(nn.Module):
    """标准的 Squeeze-and-Excitation 模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 压缩通道 -> 激活 -> 恢复通道 -> Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: 全局池化 [B, C, 1, 1]
        y = self.avg_pool(x).view(b, c)
        # Excitation: 计算通道权重 [B, C]
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 加权
        return x * y.expand_as(x)

class SelectiveE2FInjection(nn.Module):
    def __init__(self, config, event_channels, rgb_channels, inner_dim=64):
        super().__init__()
        
        # --- 1. Event 编码器 (轻量化) ---
        self.event_encoder = nn.Sequential(
            nn.Conv2d(event_channels, inner_dim, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=inner_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- 2. 空间门控 (Spatial Selection) ---
        # 负责 "Where to inject"
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # --- 3. 投影与通道选择 (Content/Channel Selection) ---
        # 负责 "What to inject"
        self.proj = nn.Conv2d(inner_dim, rgb_channels, kernel_size=1)
        
        # [新增] SE 模块，用于筛选投影后的特征通道
        self.se_block = SEBlock(rgb_channels, reduction=8) # reduction可调，通常8或16

        self._init_weights()

    def _init_weights(self):
        # 1. 投影层零初始化
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)
        
        # 2. Spatial Gate 初始化 (偏置设为负，初始关闭)
        nn.init.normal_(self.spatial_gate[-2].weight, std=0.001)
        nn.init.constant_(self.spatial_gate[-2].bias, -2.0) 

    def forward(self, x_rgb, x_events):
        # 1. 提取 Event 特征
        feat_event = self.event_encoder(x_events)

        # 2. 计算空间门控 (Spatial Attention)
        # gate: [B, 1, H, W]
        spatial_mask = self.spatial_gate(feat_event)

        # 3. 投影到 RGB 维度
        # feat_injected: [B, C_rgb, H, W]
        feat_injected = self.proj(feat_event)
        
        # 4. [关键] 应用 SE 机制 (Channel Attention)
        # 这一步会让网络根据全局信息，抑制掉那些含有噪声或无关信息的通道
        # 只有 "高质量" 的特征通道会被保留下来
        feat_injected = self.se_block(feat_injected)

        # 5. 双重加权注入
        # 最终特征 = RGB + (空间位置重要性 * 通道重要性 * Event特征)
        x_out = x_rgb + spatial_mask * feat_injected
        
        return x_out

class SelectiveE2FInjection_New(nn.Module):
    def __init__(self, config, event_channels, rgb_channels):
        super().__init__()
        
        # --- 1. Event 编码器 (轻量化) ---
        self.event_encoder = nn.Sequential(
            nn.Conv2d(event_channels, rgb_channels, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=rgb_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels, rgb_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- 2. 空间门控 (Spatial Selection) ---
        # 负责 "Where to inject"
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(rgb_channels, rgb_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # --- 3. 投影与通道选择 (Content/Channel Selection) ---        
        # [新增] SE 模块，用于筛选投影后的特征通道
        self.se_block = SEBlock(rgb_channels, reduction=2) # reduction可调，通常8或16

        self._init_weights()

    def _init_weights(self):        
        # 2. Spatial Gate 初始化 (偏置设为负，初始关闭)
        nn.init.normal_(self.spatial_gate[-2].weight, std=0.001)
        nn.init.constant_(self.spatial_gate[-2].bias, -2.0) 

    def forward(self, x_rgb, x_events):
        # 1. 提取 Event 特征
        feat_event = self.event_encoder(x_events)

        # 2. 计算空间门控 (Spatial Attention)
        # gate: [B, 1, H, W]
        spatial_mask = self.spatial_gate(feat_event)

        # 4. [关键] 应用 SE 机制 (Channel Attention)
        # 这一步会让网络根据全局信息，抑制掉那些含有噪声或无关信息的通道
        # 只有 "高质量" 的特征通道会被保留下来
        feat_injected = self.se_block(feat_event)

        # 5. 双重加权注入
        # 最终特征 = RGB + (空间位置重要性 * 通道重要性 * Event特征)
        x_out = x_rgb + spatial_mask * feat_injected
        
        return x_out
    


class GaussianBlurLayer(nn.Module):
    """ 固定参数的高斯模糊层 (Depthwise)，用于提取不同尺度的结构信息 """
    def __init__(self, channels, kernel_size=5, sigma=1.0):
        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.padding = kernel_size // 2
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('kernel', kernel)

    def _create_gaussian_kernel(self, size, sigma):
        coords = torch.arange(size).float() - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        # Depthwise kernel: [C, 1, K, K]
        return g.reshape(1, 1, size, 1) * g.reshape(1, 1, 1, size).expand(self.channels, 1, size, size)

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.padding, groups=self.channels)

class MultiScaleDoGAttention(nn.Module):
    """ 
    多尺度高斯差分注意力模块 (无损植入版)
    初始状态下 Output ≈ Input，支持热插拔和微调。
    """
    def __init__(self, channels):
        super(MultiScaleDoGAttention, self).__init__()
        self.blur_fine = GaussianBlurLayer(channels, kernel_size=3, sigma=1.0)
        self.blur_coarse = GaussianBlurLayer(channels, kernel_size=7, sigma=2.0)
        
        # 门控网络：将 DoG 特征映射为 0~1 的权重
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self._init_weights() # 执行零初始化

    def _init_weights(self):
        # 1. 遍历所有层，先进行标准的 Kaiming 初始化
        # 这样保证了第一层卷积是有“活性”的，梯度可以传进来
        for m in self.gate.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 2. 【关键修正】只对最后一层卷积进行 Zero Init + 强偏置
        # 注意：self.gate 的结构是 Conv -> ReLU -> Conv -> Sigmoid
        # 最后一层卷积是 self.gate[2] (索引从0开始: 0=Conv, 1=ReLU, 2=Conv, 3=Sigmoid)
        
        last_conv = self.gate[2] 
        
        # 权重归零：确保初始波动为 0
        nn.init.constant_(last_conv.weight, 0)
        
        # 偏置设为负大数：确保 Sigmoid 输出 ≈ 0
        nn.init.constant_(last_conv.bias, -10.0)

    def forward(self, x):
        with torch.no_grad(): # 高斯差分部分无需梯度
            dog = self.blur_fine(x) - self.blur_coarse(x)
        mask = self.gate(dog)
        return x * (1 + mask) # Residual 形式

class TimeScaleSelector(nn.Module):
    """
    针对多时间尺度输入的通道选择器。
    本质是一个轻量级的 SE-Block。
    """
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        # 如果通道数很少（比如3个），reduction设为1以保留更多信息
        reduction_channels = max(in_channels // reduction, 2) 
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduction_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
        # 初始稍微偏向中立，不要一开始就杀掉某个通道
        self._init_weights()

    def _init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # 将 Bias 设为 0，Sigmoid(0) = 0.5
                    # 这样初始权重就在 0.5 附近，梯度回传最流畅
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        # 全局池化 -> 判断当前的运动激烈程度
        y = self.avg_pool(x).view(b, c)
        # 生成各时间尺度的权重
        w = self.fc(y).view(b, c, 1, 1)
        # 加权
        return x * w