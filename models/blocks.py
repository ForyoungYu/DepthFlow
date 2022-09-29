import torch.nn as nn
from .common import dwsconv3x3_block

"""
jump connection
"""
def _make_scratch(in_shape, out_shape, groups=1, dw=False , expand=False):
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
    if dw:
        scratch.layer1_rn = dwsconv3x3_block(
            in_shape[0],
            out_shape1,
        )
        scratch.layer2_rn = dwsconv3x3_block(
            in_shape[1],
            out_shape2,
        )
        scratch.layer3_rn = dwsconv3x3_block(
            in_shape[2],
            out_shape3,
        )
        scratch.layer4_rn = dwsconv3x3_block(
            in_shape[3],
            out_shape4,
        )
    else:
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

class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, dw):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1
        if dw:
            self.conv1 = dwsconv3x3_block(features, features)
            self.conv2 = dwsconv3x3_block(features, features)
        else:
            self.conv1 = nn.Conv2d(
                features,
                features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.bn,
                groups=self.groups,
            )

            self.conv2 = nn.Conv2d(
                features,
                features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.bn,
                groups=self.groups,
            )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, dw=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn, dw)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn, dw)
        
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]
        # print("xs[0] " + str(output.shape))
        if len(xs) == 2:
            # print("xs[1] " + str(xs[1].shape)) # 4, 256, 19, 66
            res = self.resConfUnit1(xs[1])
            # print("resConv xs[1] {}".format(xs[1].shape)) # 4, 256, 19, 66
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)
        # print("Before " + str(output.shape))
        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        # print("After " + str(output.shape))
        output = self.out_conv(output)  # C/2

        return output
