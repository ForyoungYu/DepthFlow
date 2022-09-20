import torch
import torch.nn as nn
# from einops import rearrange

from .backbone.efficientformer import EfficientFormer
from .head.DPTHead import FeatureFusionBlock_custom

# 每个stage的MB输出维度
EfficientFormer_width = {
    'l1': [48, 96, 224, 448],
    'l3': [64, 128, 320, 512],
    'l7': [96, 192, 384, 768],
}

# 每个stage的MB个数
EfficientFormer_depth = {
    'l1': [3, 2, 6, 4],
    'l3': [4, 4, 12, 6],
    'l7': [6, 6, 18, 8],
}


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    """
    不改变map的形状, 只改变通道数
    """
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


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


# My Code
class EFT(nn.Module):

    def __init__(self, head, features=256, use_bn=True):
        super().__init__()

        # backbone
        self.backbone = efficientformer_l1_feat()
        # weights = 'weights/efficientformer_l1_300d.pth'
        # checkpoint = torch.load(weights, map_location='cpu')
        # self.backbone.load_state_dict(checkpoint['model'],  strict=False)

        # neck
        #? scratch是个什么神奇的东西
        self.scratch = _make_scratch(EfficientFormer_width['l1'], features)
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # head
        self.scratch.output_conv = head

    def forward(self, x):
        out = self.backbone(x)

        layer_1_rn = self.scratch.layer1_rn(out[0])
        layer_2_rn = self.scratch.layer2_rn(out[1])
        layer_3_rn = self.scratch.layer3_rn(out[2])
        layer_4_rn = self.scratch.layer4_rn(out[3])

        # print('===layer===')
        # print(layer_1_rn.shape)
        # print(layer_2_rn.shape)
        # print(layer_3_rn.shape)
        # print(layer_4_rn.shape)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # print('===paht===')
        # print(path_4.shape)
        # print(path_3.shape)
        # print(path_2.shape)
        # print(path_1.shape)

        out = self.scratch.output_conv(path_1)

        return out


class MyDepthModel(EFT):

    def __init__(self,
                 path=None,
                 non_negative=True,
                 scale=1.0,
                 shift=0.0,
                 invert=False,
                 **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert  # 反转深度

        head = nn.Sequential(
            nn.Conv2d(features,
                      features // 2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),  # 用于占位，不改变原有输入
        )

        super().__init__(head, **kwargs)

        # self.weight_init()
        if path is not None:
            self.load(path)

    def forward(self, x):
        inv_depth = super().forward(x)  # 删除通道

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth
        else:
            return inv_depth

    # def weight_init(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_normal_(m.weight)
    #         nn.init.constant_(m.bias, 0)
    #     # 也可以判断是否为conv2d，使用相应的初始化方式
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         # 是否为批归一化层
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)

    # def get_1x_lr_params(self,):  # lr/10 learning rate
    #     return self.backbone.parameters()

    # def get_10x_lr_params(self):  # lr learning rate
    #     modules = [self.backbone, self.scratch]
    #     for m in modules:
    #         yield from m.parameters()

    # @classmethod
    # def build(cls, args, **kwargs):
    #     # basemodel_name = 'tf_efficientnet_b5_ap'
    #     weights = 'weights/efficientformer_l1_300d.pth'

    #     model = cls()
    #     checkpoint = torch.hub.load(weights, pretrained=True)
    #     model.load_state_dict(checkpoint['model'])
    #     print('Done.')

    #     return model


# if __name__ == '__main__':
#     import time

#     import cv2

#     from util.io import *

#     input = torch.randn(1, 3, 224, 224)
#     img = read_image('../input/rgb_00000.jpg')
#     print('origin size' + str(img.shape))

#     #! 调整图片的大小
#     img = resize_image(img)
#     # img = torch.from_numpy(img).unsqueeze(dim=0).float()
#     # img = rearrange(img, 'b h w c -> b c h w')
#     print('resize to ' + str(img.shape))

#     model = MyDepthModel()
#     model.eval()
#     # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     # input = input.cuda()
#     # model.to(device)

#     start = time.time()
#     output = model(img)
#     end = time.time()

#     total = end - start

#     print('output shape: ' + str(output.shape))

#     write_depth('../output/rgb_00000', output, bits=1)

#     print('Runing time {:.5f} s'.format(total))
