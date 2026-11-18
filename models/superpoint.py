"""SuperPoint BN teachers and baselines.

BN portions adapted from rpautrat/SuperPoint and eric-yyjau/pytorch-superpoint.
Local modifications are covered by the project license.
"""
# MIT License (upstream portions)
# Copyright (c) 2018 Paul-Edouard Sarlin & Rémi Pautrat
# Copyright (c) 2019 Eric Jau
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from types import SimpleNamespace

class SuperPointBNNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config, input_channel=1, grid_size=8, device='cpu'):
        config = {**self.default_config, **config}
        super(SuperPointBNNet, self).__init__()

        self.backbone = VGGBackboneBN(input_channel, device=device)
        ##
        self.detector_head = DetectorHeadBN(input_channel=128,
                                          grid_size=grid_size, using_bn=True)
        self.descriptor_head = DescriptorHeadBN(input_channel=128,
                                              output_channel=config['descriptor_dim'],
                                              grid_size=grid_size, using_bn=True)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        if isinstance(x, dict):
            outputs = self.backbone(x['img'])
        else:
            outputs = self.backbone(x)
        det_outputs = self.detector_head(outputs[-1])

        prob = det_outputs['prob']

        desc_outputs = self.descriptor_head(outputs[-1])
        return {"logits": det_outputs['logits'],
                "descriptors_raw": desc_outputs['desc_raw'],
                "descriptors": desc_outputs['desc'],
                "features": outputs,
                "prob": prob}

class VGGBackboneBN(nn.Module):
    """vgg backbone to extract feature
    Note:set eps=1e-3 for BatchNorm2d to reproduce results
         of pretrained model `superpoint_bn.pth`
    """
    def __init__(self, input_channel=1, device='cpu'):
        super(VGGBackboneBN, self).__init__()
        self.device = device
        channels = [64,64,64,64,128,128,128,128]

        self.block1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, channels[0], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[0]),
        )

        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[1]),
        )
        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[2]),
        )
        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[3]),
        )
        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[4]),
        )
        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[5]),
        )
            # block 3
        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[6]),
        )
        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[6], channels[7], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[7]),
        )

        self.pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        outputs = []
        out = self.block1_1(x)
        out = self.block1_2(out)
        outputs.append(out)
        out = self.pooling(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        outputs.append(out)
        out = self.pooling(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        outputs.append(out)
        out = self.pooling(out)
        out = self.block4_1(out)
        feat_map = self.block4_2(out)
        outputs.append(feat_map)

        return outputs

class DetectorHeadBN(torch.nn.Module):
    def __init__(self, input_channel, grid_size, using_bn=True):
        super(DetectorHeadBN, self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn
        ##
        self.convPa = torch.nn.Conv2d(input_channel, 256, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.convPb = torch.nn.Conv2d(256, pow(grid_size, 2)+1, kernel_size=1, stride=1, padding=0)

        self.bnPa,self.bnPb = None,None
        if using_bn:
            self.bnPa = torch.nn.BatchNorm2d(256)
            self.bnPb = torch.nn.BatchNorm2d(pow(grid_size, 2)+1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = None
        if self.using_bn:
            out = self.bnPa(self.relu(self.convPa(x)))
            out = self.bnPb(self.convPb(out))  #(B,65,H,W)
        else:
            out = self.relu(self.convPa(x))
            out = self.convPb(out)  # (B,65,H,W)

        prob = self.softmax(out)
        prob = prob[:, :-1, :, :]  # remove dustbin,[B,64,H,W]
        # Reshape to get full resolution heatmap.
        prob = torch.nn.functional.pixel_shuffle(prob, self.grid_size)  # [B,1,H*8,W*8]
        prob = prob.squeeze(dim=1)#[B,H,W]

        return {'logits':out, 'prob':prob}

class DescriptorHeadBN(torch.nn.Module):
    def __init__(self, input_channel, output_channel, grid_size, using_bn=True):
        super(DescriptorHeadBN, self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn

        self.convDa = torch.nn.Conv2d(input_channel, 256, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.convDb = torch.nn.Conv2d(256, output_channel, kernel_size=1, stride=1, padding=0)

        self.bnDa, self.bnDb = None, None
        if using_bn:
            self.bnDa = torch.nn.BatchNorm2d(256)
            self.bnDb = torch.nn.BatchNorm2d(output_channel)

    def forward(self, x):
        out = None
        if self.using_bn:
            out = self.bnDa(self.relu(self.convDa(x)))
            out = self.bnDb(self.convDb(out))
        else:
            out = self.relu(self.convDa(x))
            out = self.convDb(out)

        # out_norm = torch.norm(out, p=2, dim=1)  # Compute the norm.
        # out = out.div(torch.unsqueeze(out_norm, 1))  # Divide by norm to normalize.

        # TODO: here is different with tf.image.resize_bilinear
        desc = F.interpolate(out, scale_factor=self.grid_size, mode='bilinear',align_corners=False)
        desc = F.normalize(desc, p=2, dim=1)  # normalize by channel
        out = F.normalize(out, p=2, dim=1)

        return {'desc_raw': out, 'desc': desc}

####
class SuperPointNet_gauss2(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, subpixel_channel=1):
        super(SuperPointNet_gauss2, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)

        self.relu = torch.nn.ReLU(inplace=True)

        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)



    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        outputs = [x1, x2, x3, x4]

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))

        prob = F.softmax(semi, dim=1)
        prob = prob[:, :-1, :, :]  # remove dustbin,[B,64,H,W]
        # Reshape to get full resolution heatmap.
        prob = torch.nn.functional.pixel_shuffle(prob, 8)  # [B,1,H*8,W*8]
        prob = prob.squeeze(dim=1)#[B,H,W]

        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))

        desc_expand = F.interpolate(desc, scale_factor=8, mode='bilinear',align_corners=False)
        desc_expand = F.normalize(desc_expand, p=2, dim=1)  #normalize by channel
        desc = F.normalize(desc, p=2, dim=1)



        output = {"logits": semi,
                "descriptors_raw": desc,
                "descriptors": desc_expand,
                "features": outputs,
                "prob": prob}

        return output

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

####
class SuperPointBatchNormed(nn.Module):
    default_conf = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
    }

    def __init__(self, **conf):
        super().__init__()
        conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**conf)
        self.stride = 2 ** (len(self.conf.channels) - 2)
        channels = [1, *self.conf.channels[:-1]]

        self.backbone = nn.Sequential(
            # Block 0: 输入 -> 64 (Key: backbone.0.*)
            nn.Sequential(
                VGGBlock(1, 64, 3),
                VGGBlock(64, 64, 3),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),

            # Block 1: 64 -> 64 (Key: backbone.1.*)
            nn.Sequential(
                VGGBlock(64, 64, 3),
                VGGBlock(64, 64, 3),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),

            # Block 2: 64 -> 128 (Key: backbone.2.*)
            nn.Sequential(
                VGGBlock(64, 128, 3),
                VGGBlock(128, 128, 3),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),

            # Block 3: 128 -> 128 (Key: backbone.3.*)
            # 注意：最后一个 Block 没有 MaxPool，这正好符合你的 logic
            nn.Sequential(
                VGGBlock(128, 128, 3),
                VGGBlock(128, 128, 3),
            ),
        )

        c = self.conf.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.conf.descriptor_dim, 1, relu=False),
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


class VGGBlock(nn.Sequential):
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

