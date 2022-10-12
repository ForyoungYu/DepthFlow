"""Module containing all components necessary for creating a MoCoVit network.

Based on https://arxiv.org/abs/2205.12635v1. Seriously, read the paper!

Variable names try to follow ghostnet.pytorch repository.
"""
import torch
import torch.nn as nn
from ..modules import ghost_net


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
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dw_conv = ghost_net.depthwise_conv(inp, inp)  # 需要自行实现
        self.ghost_module = ghost_net.GhostModule(inp, oup)  # 需要自行实现

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Calculate mobile self-attention. See Equation 3.

        Args:
            v (torch.Tensor): Input vector representing the 'Value' in Q, K, V.

        Returns:
            torch.Tensor: Result of mobile self-attention.
        """
        out = torch.matmul(self.attend(torch.matmul(v, v.transpose(-1, -2)) * self.scale), v)
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
            ghost_net.GhostModule(inp, hidden_dim),
            ghost_net.SELayer(hidden_dim),
            ghost_net.GhostModule(hidden_dim, oup),
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
        mosa_out = self.mosa(x) + x
        moffn_out = self.moffn(mosa_out)

        return moffn_out + mosa_out


class MoCoViT(nn.Module):
    """Mobile Convolutional Transformer network"""
    def __init__(self, input_channel: int = 160, exp_size: int = 960, num_blocks: int = 4, width_mult: int = 1):
        """Initialize MoCoViT module.

        Args:
            input_channel (int, optional): Number of input channels. Defaults to 160.
            exp_size (int, optional): Expansion size. Defaults to 960.
            num_blocks (int, optional): Number of MoTBlocks. Defaults to 4.
            width_mult (int, optional): Width multiplier. Defaults to 1.
        """
        super(MoCoViT, self).__init__()
        self.input_channel = input_channel
        self.exp_size = exp_size
        self.num_blocks = num_blocks
        self.width_mult = width_mult

        self._ghost_net = ghost_net.ghost_net()
        self.ghost_blocks = nn.Sequential(*list(self._ghost_net.children())[0][:13])
        # self.squeeze = self._ghost_net.squeeze

        mtb_layers = []
        output_channel = ghost_net._make_divisible(self.input_channel * self.width_mult, 4)
        hidden_channel = ghost_net._make_divisible(self.exp_size * self.width_mult, 4)
        for _ in range(self.num_blocks):
            mtb_layers.append(MoTBlock(self.input_channel, hidden_channel, output_channel))

        self.motblocks = nn.Sequential(*mtb_layers)
        # self.classifier = self._ghost_net.classifier

        # Decoder

        # Depth Head
        # self.depth_head = nn.Sequential(
        #     nn.Conv2d(features[0],
        #               features[0] // 2,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),  # 64 -> 32
        #     Interpolate(scale_factor=2,
        #                 mode="bilinear",
        #                 align_corners=align_corners),
        #     nn.Conv2d(features[0] // 2, 32, kernel_size=3, stride=1,
        #               padding=1),  # 32 -> 32
        #     self.scratch.activation,
        #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),  # 32 -> 1
        #     nn.ReLU(True) if non_negative else nn.Identity(),
        #     nn.Identity())
        
        # self.pre_depth = conv1x1(256, 256, groups=256, bias=False)
        # self.depth = conv3x3(256, 1, bias=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MoCoViT.

        Args:
            x (torch.Tensor): Input tensor (batch of images).

        Returns:
            torch.Tensor: Output tensor (classifications).
        """
        # x = self.squeeze(self.motblocks(self.ghost_blocks(x)))

        x = self.ghost_blocks(x)
        x = self.motblocks(x)

        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        return x
