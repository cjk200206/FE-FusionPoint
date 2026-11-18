import torch
from torch import nn

from models.backbones.vgg import VggBlock, VGGBN
from models.backbones.maxvit import MaxVitAttentionPairCl
from models.backbones.maxvit_backbone.layers.maxvit.maxvit import nhwC_2_nChw, nChw_2_nhwC
from models.backbones.maxvit_backbone.fusion_blocks import FusionBlock

# For TensorRT, since 'prob = prob[:, :-1, :, :]' does not work
class RemoveLastChannel(nn.Module):
    def __init__(self, num_channels: int = 65):
        super().__init__()
        # register constant buffer with selected indices [0, ..., 63]
        self.register_buffer("indices", torch.arange(num_channels - 1))

    def forward(self, x):
        # x: (N, C, H, W), indices: (64,)
        return torch.index_select(x, dim=1, index=self.indices)

class DetectorHead(nn.Module):
    def __init__(self, config, input_channels=128, grid_size=8, norm = 'bn', use_spbn=False):
        super().__init__()
        self.grid_size = grid_size
        self.remove_dustbin = RemoveLastChannel(65)
        if use_spbn:
            self.layers = nn.Sequential(
                VGGBN(input_channels, 256, 3),
                VGGBN(256, 1+pow(self.grid_size, 2), 1, relu=False)
                )
        else:
            self.layers = nn.Sequential(
                VggBlock(input_channels, 256, 3),
                VggBlock(256, 1+pow(self.grid_size, 2), 1, activate=False)
                )

    def forward(self, x):
        x = self.layers(x)

        # PyTorch default is channels first [B, C, H, W]
        prob = torch.nn.functional.softmax(x, dim=1)

        # Remove dustbin for "no interest point"
        # prob = prob[:, :-1, :, :]
        # TensorRT fails on the line above
        prob = self.remove_dustbin(prob)  # (N, 64, H, W)

        prob = torch.nn.functional.pixel_shuffle(prob, self.grid_size)
        prob = torch.squeeze(prob, dim=1)

        return x, prob

class DetectorHeadFullRes(nn.Module):
    def __init__(self, config, input_channels=128, norm = 'bn'):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential(
            VggBlock(input_channels, 256, 3),
            VggBlock(256, 1, 1, activate=False)
            )

    def forward(self, x):
        x = self.layers(x)

        x = torch.squeeze(x, dim=1)
        prob = torch.nn.functional.sigmoid(x)

        return x, prob
    
class DescriptorHead(nn.Module):
    def __init__(self, config, input_channels=128, grid_size=8, descriptor_size=256, interpolate=True, norm = 'bn', use_spbn=False):
        super().__init__()
        self.grid_size = grid_size
        self.descriptor_size = descriptor_size
        self.interpolate = interpolate
        if use_spbn:
            self.layers = nn.Sequential(
                VGGBN(input_channels, 256, 3),
                VGGBN(256, self.descriptor_size, 1, relu=False)
                )
        else:
            # self.layers = nn.Sequential(
            #     VggBlock(input_channels, 256, 3, norm=norm),
            #     VggBlock(256, self.descriptor_size, 1, activate=False, norm=norm)
            #         )
            self.layers = nn.Sequential(
                VggBlock(input_channels, 256, 3),
                VggBlock(256, self.descriptor_size, 1, activate=False)
                    )


    def forward(self, x):
        x = self.layers(x)

        if self.interpolate:
            # PyTorch default is channels first [B, C, H, W]
            desc_raw_shape = x.shape[2:]
            input_shape = [self.grid_size*desc_raw_shape[0], self.grid_size*desc_raw_shape[1]]
            desc = torch.nn.functional.interpolate(x, scale_factor=self.grid_size, mode="bilinear")
            # desc = torch.nn.functional.normalize(desc, dim=1)
            # safe, numerically stable version for TensorRT
            norm = (desc * desc).sum(dim=1, keepdim=True).sqrt()
            desc = desc / (norm + 1e-6)
        else:
            desc = torch.nn.functional.normalize(x, dim=1)

        return x, desc
    


    
class DetectorHeadFusion(nn.Module):
    def __init__(self, config, input_channels=128, grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        self.remove_dustbin = RemoveLastChannel(65)
        self.layers = nn.Sequential(
            VggBlock(input_channels, 256, 3),
            VggBlock(256, 1+pow(self.grid_size, 2), 1, activate=False)
            )
        config_fuse = config["fusion_config"]
        config_fuse["attention"]["dim_head"] = 1 

        self.fusion_layer = FusionBlock(config_fuse, dim = 1+pow(self.grid_size, 2))
        self.add_conv = VggBlock(1+pow(self.grid_size, 2), 1+pow(self.grid_size, 2), 1, activate=False)

    def forward(self, input_dict):
        x = input_dict["fusion_feature"]
        fusion_dict = {}
        fusion_dict["event_features"] = input_dict["E_keypoint"]
        fusion_dict["frame_features"] = input_dict["F_keypoint"]

        x = self.layers(x)
        fused_keypoint = self.fusion_layer(fusion_dict)
        x = self.add_conv(x+fused_keypoint)

        # PyTorch default is channels first [B, C, H, W]
        prob = torch.nn.functional.softmax(x, dim=1)
        prob = self.remove_dustbin(prob)  # (N, 64, H, W)

        prob = torch.nn.functional.pixel_shuffle(prob, self.grid_size)
        prob = torch.squeeze(prob, dim=1)

        return x, prob

class DescriptorHeadFusion(nn.Module):
    def __init__(self, config, input_channels=128, grid_size=8, descriptor_size=256, interpolate=True):
        super().__init__()
        self.grid_size = grid_size
        self.descriptor_size = descriptor_size
        self.interpolate = interpolate
        self.layers = nn.Sequential(
            VggBlock(input_channels, 256, 3),
            VggBlock(256, self.descriptor_size, 1, activate=False)
                )
        config_fuse = config["fusion_config"]
        config_fuse["attention"]["dim_head"] = 1

        self.fusion_layer = FusionBlock(config_fuse, dim = self.descriptor_size)
        self.add_conv = VggBlock(self.descriptor_size, self.descriptor_size, 1, activate=False)

    def forward(self, input_dict):
        x = input_dict["fusion_feature"]
        fusion_dict = {}
        fusion_dict["event_features"] = input_dict["E_desc"]
        fusion_dict["frame_features"] = input_dict["F_desc"]

        x = self.layers(x)
        fused_desc = self.fusion_layer(fusion_dict)
        x = self.add_conv(x+fused_desc)

        if self.interpolate:
            # PyTorch default is channels first [B, C, H, W]
            desc_raw_shape = x.shape[2:]
            input_shape = [self.grid_size*desc_raw_shape[0], self.grid_size*desc_raw_shape[1]]
            desc = torch.nn.functional.interpolate(x, scale_factor=self.grid_size, mode="bilinear")

            norm = (desc * desc).sum(dim=1, keepdim=True).sqrt()
            desc = desc / (norm + 1e-6)
        else:
            desc = torch.nn.functional.normalize(x, dim=1)

        return x, desc