import torch
import torch.nn as nn

from .efficientformer import EfficientFormer
from ..modules.base_model import BaseModel
from .blocks import FeatureFusionBlock_custom, Interpolate, _make_scratch, _make_crp
"""
添加了MoCoVit模块和dwconv
修改FeatureFusionBlock_custom使得其可以自由设定维度
"""

# 每个stage的MB输出维度
features = {
    'l1': [48, 96, 224, 448],
    'l3': [64, 128, 256, 512],
    'l7': [96, 192, 384, 768],
    'custom': [64, 128, 256, 512]
}

# 每个stage的MB个数
depth = {
    'l1': [3, 2, 6, 4],
    'l3': [4, 4, 12, 6],
    'l7': [6, 6, 18, 8],
    'custom': [4, 4, 12, 6]
}


class EFTv2_1(BaseModel):
    def __init__(self,
                 path=None,
                 layers=depth['custom'],
                 features=features['custom'],
                 non_negative=True,
                 channels_last=False,
                 use_bn=True,
                 align_corners=True):
        if path:
            print("Loading weights: ", path)

        super(EFTv2_1, self).__init__()

        self.channels_last = channels_last

        # Backbone
        self.backbone = EfficientFormer(layers=layers,
                         embed_dims=features,
                         downsamples=[True, True, True, True],
                         layer_scale_init_value=1e-6,
                         fork_feat=True,
                         act_layer=nn.ReLU)

        use_dw = True
        # Neck
        # self.scratch = _make_scratch(features, features, dw=use_dw)
        self.scratch = _make_crp(features, features, stages=4)

        # Fusion
        self.scratch.activation = nn.ReLU(False)
        self.scratch.refinenet4 = FeatureFusionBlock_custom(
            features[3],
            features[2],
            self.scratch.activation,
            deconv=False,
            bn=use_bn,
            dw=use_dw,
            align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(
            features[2],
            features[1],
            self.scratch.activation,
            deconv=False,
            bn=use_bn,
            dw=use_dw,
            align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(
            features[1],
            features[0],
            self.scratch.activation,
            deconv=False,
            bn=use_bn,
            dw=use_dw,
            align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(
            features[0],
            features[0],
            self.scratch.activation,
            deconv=False,
            bn=use_bn,
            dw=use_dw,
            align_corners=align_corners)

        # Head
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features[0],
                      features[0] // 2,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # 64 -> 32
            Interpolate(scale_factor=2,
                        mode="bilinear",
                        align_corners=align_corners),
            nn.Conv2d(features[0] // 2, 32, kernel_size=3, stride=1,
                      padding=1),  # 32 -> 32
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),  # 32 -> 1
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity())

        if path is not None:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        if self.channels_last == True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)
        out = self.backbone(x)
        # print("=== out ===")
        # print(out[0].shape)
        # print(out[1].shape)
        # print(out[2].shape)
        # print(out[3].shape)

        layer_1_rn = self.scratch.layer1_rn(out[0])
        layer_2_rn = self.scratch.layer2_rn(out[1])
        layer_3_rn = self.scratch.layer3_rn(out[2])
        layer_4_rn = self.scratch.layer4_rn(out[3])
        # print('=== layers ===')
        # print(layer_1_rn.shape)
        # print(layer_2_rn.shape)
        # print(layer_3_rn.shape)
        # print(layer_4_rn.shape)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        # print('=== pathes ===')
        # print("path_4 " + str(path_4.shape))
        # print("path3 " + str(path_3.shape))
        # print("path2 " + str(path_2.shape))
        # print("path1 " + str(path_1.shape))

        out = self.scratch.output_conv(path_1)

        # return torch.squeeze(out, dim=1)
        return out

