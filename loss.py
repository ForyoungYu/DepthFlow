import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.eps = 0.001  # avoid grad explode

    def forward(self, input, target, mask=None, interpolate=True):
        # n, c, h, w = target.shape
        if interpolate:
            # interpolate input shape: n, c, h, w
            input = nn.functional.interpolate(input,
                                              target.shape[-2:],
                                              mode='bilinear',
                                              align_corners=True)

        if mask is not None:  # 对mask为True的值进行保留，并转换成一维数据
            input = input[mask]
            target = target[mask]
        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)  # 1/T
        # Dg = norm * torch.sum(g**2) - (0.85*(norm**2)) * (torch.sum(g))**2  # Dg >=0
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        # print("Dg: {}".format(Dg))
        return 10 * torch.sqrt(Dg)


class SigLoss(nn.Module):
    """SigLoss.
        We adopt the implementation in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_.
    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """
    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 min_depth=0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100,
                 interpolate=True):
        super(SigLoss, self).__init__()
        self.name = 'SILog'
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.interpolate = interpolate

        self.eps = 0.001  # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.interpolate:
            # interpolate input shape: n, c, h, w
            input = nn.functional.interpolate(input,
                                              target.shape[-2:],
                                              mode='bilinear',
                                              align_corners=True)
        if self.valid_mask:
            valid_mask = target > self.min_depth
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > self.min_depth,
                                               target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""

        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt)
        return loss_depth
