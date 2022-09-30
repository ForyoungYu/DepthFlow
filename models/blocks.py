import math

import torch
import torch.nn as nn

from .common import SEBlock, conv1x1_block, dwconv3x3_block


# 反卷积
class InvertedResidual(nn.Module):
    """ InvertedResidual """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim,
                          hidden_dim,
                          3,
                          stride,
                          1,
                          groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim,
                          hidden_dim,
                          3,
                          stride,
                          1,
                          groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class GhostConvBlock(nn.Module):
    """
    GhostNet specific convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(GhostConvBlock, self).__init__()
        main_out_channels = math.ceil(0.5 * out_channels)  # ceil: 向上取整
        cheap_out_channels = out_channels - main_out_channels

        self.main_conv = conv1x1_block(in_channels=in_channels,
                                       out_channels=main_out_channels,
                                       activation=activation)
        self.cheap_conv = dwconv3x3_block(in_channels=main_out_channels,
                                          out_channels=cheap_out_channels,
                                          activation=activation)

    def forward(self, x):
        x = self.main_conv(x)
        y = self.cheap_conv(x)
        return torch.cat((x, y), dim=1)


"""Module containing all components necessary for creating a MoCoVit network.

Based on https://arxiv.org/abs/2205.12635v1. Seriously, read the paper!

Variable names try to follow ghostnet.pytorch repository.
"""


class MoSA(nn.Module):
    """Mobile Self-Attention Module"""
    def __init__(self, inp: int, oup: int):
        """Initialize a MoSA module.

        Args:
            inp (int): Input channel size.
            oup (int): Output channel size.
        """
        super(MoSA, self).__init__()
        self.dim_head = 64
        self.scale = self.dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dw_conv = dwconv3x3_block(inp, inp)
        self.ghost_module = GhostConvBlock(inp, oup)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Calculate mobile self-attention. See Equation 3.

        Args:
            v (torch.Tensor): Input vector representing the 'Value' in Q, K, V.

        Returns:
            torch.Tensor: Result of mobile self-attention.
        """
        out = torch.matmul(
            self.attend(torch.matmul(v, v.transpose(-1, -2)) * self.scale), v)
        out = out + self.dw_conv(v)

        return self.ghost_module(out)


class MoFFN(nn.Module):
    """Mobile Feed Forward Network"""
    def __init__(self, inp: int, hidden_dim: int, oup: int):
        """Initialize a MoFFN module.

        Args:
            inp (int): Input channel size.
            hidden_dim (int): Hidden dimension size.
            oup (int): Output channel size.
        """
        super(MoFFN, self).__init__()
        self.ffn = nn.Sequential(
            GhostConvBlock(inp, hidden_dim),
            SEBlock(
                hidden_dim),  # hideen_dim must less than SENlock reduction.
            GhostConvBlock(hidden_dim, oup),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MoFFN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.ffn(x) + x


class MoTBlock(nn.Module):
    """Mobile Transformer Block"""
    def __init__(self, inp: int, hidden_dim: int, oup: int):
        """Initialize a MoTBlock module.

        Args:
            inp (int): Input channel size.
            hidden_dim (int): Hidden dimension size.
            oup (int): Output channel size.
        """
        super(MoTBlock, self).__init__()
        self.mosa = MoSA(inp, oup)
        self.moffn = MoFFN(inp, hidden_dim, oup)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MoTBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # print('x: {}'.format(x.shape))
        # print(self.mosa(x).shape)
        mosa_out = self.mosa(x) + x
        moffn_out = self.moffn(mosa_out)

        return moffn_out + mosa_out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
