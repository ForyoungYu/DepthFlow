"""Module containing all components necessary for creating a MoCoVit network.

Based on https://arxiv.org/abs/2205.12635v1. Seriously, read the paper!

Variable names try to follow ghostnet.pytorch repository.
"""
import torch
import torch.nn as nn
from models.modules import MoTBlock
from models.modules import ghost_net


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
        self.squeeze = self._ghost_net.squeeze

        mtb_layers = []
        for _ in range(self.num_blocks):
            output_channel = ghost_net._make_divisible(self.input_channel * self.width_mult, 4)
            hidden_channel = ghost_net._make_divisible(self.exp_size * self.width_mult, 4)
            mtb_layers.append(MoTBlock(self.input_channel, hidden_channel, output_channel))

        self.motblocks = nn.Sequential(*mtb_layers)
        self.classifier = self._ghost_net.classifier


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MoCoViT.

        Args:
            x (torch.Tensor): Input tensor (batch of images).

        Returns:
            torch.Tensor: Output tensor (classifications).
        """
        x = self.squeeze(self.motblocks(self.ghost_blocks(x)))
        x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        return x
