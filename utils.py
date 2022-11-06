import base64
import math
import re
from io import BytesIO

import matplotlib.cm
import numpy as np
import torch
from PIL import Image
import requests


def send_massage(token, title, name, msg:dict):
    content ="a1: {:.3f} a2: {:.3f} a3: {:.3f} | Abs Rel.: {:.3f} Sq Rel.: {:.3f} | \
        RMSE: {:.3f} RMSELog: {:.3f} | SiLog: {:.3f} Log10: {:.3f}".format(\
            msg['a1'], msg['a2'], msg['a3'], \
            msg['abs_rel'], msg['sq_rel'], \
            msg['rmse'], msg['rmse_log'], \
            msg['silog'], msg['log_10'])
    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                        json={
                            "token": token,
                            "title": title,
                            "name": name,
                            "content": content
                        })
    # print(resp.content.decode())

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def denormalize(x, device='cpu'):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    return x * std + mean


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}


def colorize(value, vmin=10, vmax=1000, cmap='magma_r'):
    value = value.detach().cpu().numpy()[0, :, :]
    # value = np.log10(value)

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    value = value.squeeze(axis=0)
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # nxmx4

    img = value[:, :, :3]

    # return img.transpose((2, 0, 1))
    return img


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate(gt, pred):
    if gt.shape[0] == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    if np.isnan(silog):
        silog = 0
        
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel

def metrics(gt, pred, min_depth=1e-3, max_depth=80):
    mask_1 = gt > min_depth
    mask_2 = gt < max_depth
    mask = np.logical_and(mask_1, mask_2)

    gt = gt[mask]
    pred = pred[mask]

    a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = calculate(gt, pred)

    return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel

def eval_metrics(gt, pred, min_depth=1e-3, max_depth=80):
    mask_1 = gt > min_depth
    mask_2 = gt < max_depth
    mask = np.logical_and(mask_1, mask_2)

    gt = gt[mask]
    pred = pred[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def FPS(model,input_size=224):
    import time
    model.eval() 
    total=0
    for x in range(0,200):
        input = torch.randn(1, 3, input_size, input_size).cuda()
        with torch.no_grad():
            a = time.perf_counter()
            output = model.forward(input)
            torch.cuda.synchronize()
            b = time.perf_counter()
            total+=b-a
    print('FPS:', str(200/total))
    print('ms:', str(1000*total/200))

def flops(model, input_size):
    from thop import profile, clever_format
    input = torch.rand(1, 3, input_size, input_size).cuda()
    flops, params = profile(model.cuda(), inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops: {}, params: {}'.format(flops, params))

##################################### Demo Utilities ############################################
def b64_to_pil(b64string):
    image_data = re.sub('^data:image/.+;base64,', '', b64string)
    # image = Image.open(cStringIO.StringIO(image_data))
    return Image.open(BytesIO(base64.b64decode(image_data)))


# Compute edge magnitudes
from scipy import ndimage


def edges(d):
    dx = ndimage.sobel(d, 0)  # horizontal derivative
    dy = ndimage.sobel(d, 1)  # vertical derivative
    return np.abs(dx) + np.abs(dy)


class PointCloudHelper():
    def __init__(self, width=640, height=480):
        self.xx, self.yy = self.worldCoords(width, height)

    def worldCoords(self, width=640, height=480):
        hfov_degrees, vfov_degrees = 57, 43
        hFov = math.radians(hfov_degrees)
        vFov = math.radians(vfov_degrees)
        cx, cy = width / 2, height / 2
        fx = width / (2 * math.tan(hFov / 2))
        fy = height / (2 * math.tan(vFov / 2))
        xx, yy = np.tile(range(width), height), np.repeat(range(height), width)
        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        return xx, yy

    def depth_to_points(self, depth):
        depth[edges(depth) > 0.3] = np.nan  # Hide depth edges
        length = depth.shape[0] * depth.shape[1]
        # depth[edges(depth) > 0.3] = 1e6  # Hide depth edges
        z = depth.reshape(length)

        return np.dstack((self.xx * z, self.yy * z, z)).reshape((length, 3))

#####################################################################################################
