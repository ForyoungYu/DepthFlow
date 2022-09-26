import torch
import torch.nn as nn
# from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence

# Adabins 
class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        o = 1e-8
        # n, c, h, w = target.shape
        if interpolate:
            # interpolate input shape: n, c, h, w
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:  # 对mask为True的值进行保留，并转换成一维数据
            input = input[mask]
            target = target[mask]
        g = torch.log(input + o) - torch.log(target + o)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)  # 1/T
        # Dg = norm * torch.sum(g**2) - (0.85*(norm**2)) * (torch.sum(g))**2  # Dg >=0
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)  # Dg >= 0
        # print("Dg: {}".format(Dg))
        return 10 * torch.sqrt(Dg)


class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss

# if __name__ == '__main__':
#     input = torch.tensor([[.1, .1, .9], [.5, .5, .5], [.1, .1, .9]]).unsqueeze(0).unsqueeze(0)
#     # input = torch.randint(1, 10, (3, 3)).unsqueeze(0).unsqueeze(0) / 10
#     # input = torch.randn((3, 3)).unsqueeze(0).unsqueeze(0)
#     target = torch.tensor([[.2,.1,.8], [.5, .4, 0.2], [0.2, 0.3, 0.6]]).unsqueeze(0).unsqueeze(0)
#     mask = torch.tensor([[True, True, False], [True, True, False], [False, True, True]]).unsqueeze(0).unsqueeze(0)

#     print(input)
#     print(target)
#     print(mask)

#     loss = SILogLoss()
#     out = loss(input, target, mask=mask)
#     print(out)
