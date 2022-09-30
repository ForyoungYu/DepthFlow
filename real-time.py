import time
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from models import EFT, EFTv2, MidasNet, MidasNet_small
from models.midas.midas_net_custom import MidasNet_small

def FPS(model,input_size=224):
    import time
    model.eval()

    # for x in range(0,200):
    #     input = torch.randn(1, 3, input_size, input_size).cuda()
    #     with torch.no_grad():
    #         # import ipdb;ipdb.set_trace()
    #         output = model.forward(input)
        
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

def flops(model,input_size):
    from thop import profile, clever_format
    input = torch.rand(1, 3, input_size, input_size).cuda()
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops: {}, params: {}'.format(flops, params))


def transform(img):
    # 请根据实际情况来设定transform
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 544), interpolation=2)
    ])
    img = transf(img)  # (cxhxw)
    return img.unsqueeze(0)  # (bxcxhxw)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cpu')  # Force CPU
print('Device: {}'.format(device))

# Initial model
# model = EFTv2()
model = EFTv2()
# model = EFT(model='l3')
FPS(model.cuda(), 224)
flops(model.cuda(), 224)
exit()

# Load pretrained model
# ckpt = 'checkpoints\EFT_l3_kitti_27-Sep_16-06-nodebs4-tep100-lr0.000357-wd0.1_best.pt'
# model.load_state_dict(torch.load(ckpt), strict=False)
model.to(device)
model.eval()

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, img = cap.read()

    start = time.time()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply input transforms
    input_batch = transform(img).to(device)
    
    # Prediction and resize to original resolution
    with torch.no_grad():
        pred = model(input_batch)
        
        if len(pred.shape) == 4:
            pred = pred.squeeze(0)

        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = pred.cpu().numpy()
    depth_map = cv2.normalize(
        depth_map,
        None,
        0,
        1,
        norm_type=cv2.NORM_MINMAX,
    )

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    print("FPS: %.2f" % fps)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)  # Color mode

    cv2.putText(img, f'FPS: %.2f' % fps, (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 2)
    cv2.imshow('Image', img)
    cv2.imshow('Depth Map', depth_map)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
