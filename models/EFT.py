import torch
import torch.nn as nn

from .backbone.efficientformer import EfficientFormer
from .base_model import BaseModel
from .blocks import FeatureFusionBlock_custom, Interpolate, _make_scratch

# 每个stage的MB输出维度
EfficientFormer_width = { # x2
    # 'l1': [48, 96, 224, 448],
    'l1': [48, 96, 192, 384],  # x2
    'l3': [64, 128, 320, 512],  # x2
    'l7': [96, 192, 384, 768],  # x2
}

# 每个stage的MB个数
EfficientFormer_depth = {
    'l1': [3, 2, 6, 4],
    'l3': [4, 4, 12, 6],
    'l7': [6, 6, 18, 8],
}


class efficientformer_l1_feat(EfficientFormer):
    def __init__(self, **kwargs):
        super().__init__(layers=EfficientFormer_depth['l1'],
                         embed_dims=EfficientFormer_width['l1'],
                         downsamples=[True, True, True, True],
                         fork_feat=True,
                         vit_num=1,
                         **kwargs)


class efficientformer_l3_feat(EfficientFormer):
    def __init__(self, **kwargs):
        super().__init__(layers=EfficientFormer_depth['l3'],
                         embed_dims=EfficientFormer_width['l3'],
                         downsamples=[True, True, True, True],
                         fork_feat=True,
                         vit_num=4,
                         **kwargs)


class efficientformer_l7_feat(EfficientFormer):
    def __init__(self, **kwargs):
        super().__init__(layers=EfficientFormer_depth['l7'],
                         embed_dims=EfficientFormer_width['l7'],
                         downsamples=[True, True, True, True],
                         layer_scale_init_value=1e-6,
                         fork_feat=True,
                         vit_num=8,
                         **kwargs)


class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups):
        super(DSConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False
        )

    def forward(self, input):
        out = self.conv(input)
        return out


# My Code
class EFT(BaseModel):
    def __init__(self, path=None, features=64, model='l1', non_negative=True, channels_last=False, use_bn=True, align_corners=True, 
                 blocks={'expand': True}):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 64.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        if path:
            print("Loading weights: ", path)

        super(EFT, self).__init__()

        self.channels_last = channels_last
        self.blocks = blocks
        self.model = model

        self.groups = 1

        # 数越大越深
        features1 = features
        features2 = features
        features3 = features
        features4 = features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1 = features
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8

        # EfficientFormer Backbone
        if model == 'l1':
            self.model = efficientformer_l1_feat()
        elif model == 'l3':
            self.model = efficientformer_l3_feat()
        elif model == 'l7':
            self.model = efficientformer_l7_feat()
        else:
            print("Invalid backbone")
            assert False
        
        # Neck
        self.scratch = _make_scratch(EfficientFormer_width[model], features, expand=self.expand)

        # Fusion
        self.scratch.activation = nn.ReLU(False)
        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=use_bn, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=use_bn, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=use_bn, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=use_bn, align_corners=align_corners)

        # Head
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),  # 64 -> 32
            Interpolate(scale_factor=2, mode="bilinear", align_corners=align_corners),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),  # 32 -> 32
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),  # 32 -> 1
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity()
        )

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

        out = self.model(x)
        # print("=== out ===")
        # print(out[0].shape)
        # print(out[1].shape)
        # print(out[2].shape)
        # print(out[3].shape)
        
        # print('=== layers ===')
        layer_1_rn = self.scratch.layer1_rn(out[0])
        # print(layer_1_rn.shape)
        layer_2_rn = self.scratch.layer2_rn(out[1])
        # print(layer_2_rn.shape)
        layer_3_rn = self.scratch.layer3_rn(out[2])
        # print(layer_3_rn.shape)
        layer_4_rn = self.scratch.layer4_rn(out[3])
        # print(layer_4_rn.shape)


        # print('=== pathes ===')
        path_4 = self.scratch.refinenet4(layer_4_rn)
        # print(path_4.shape)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)  # 512 256
        # print(path_3.shape)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        # print(path_2.shape)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        # print(path_1.shape)


        out = self.scratch.output_conv(path_1)

        return out

if __name__ == '__main__':
    import time
    from util.io import *

    # input = torch.randn(1, 3, 224, 224)
    img = read_image('../input/1.png')
    print('origin size' + str(img.shape))

    #! 调整图片的大小
    img = resize_image(img)
    print('resize to ' + str(img.shape))

    model = EFT(channels_last=True)
    model.eval()
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # input = input.cuda()
    # model.to(device)

    start = time.time()
    output = model(img)
    end = time.time()

    total = end - start

    print('output shape: ' + str(output.shape))

    write_depth('../output/out.png', output, bits=1)

    print('Runing time {:.5f} s'.format(total))
