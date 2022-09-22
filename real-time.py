import time
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from models.EFT import EFT


SHOW_WINDOW = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cpu')  # Force CPU
print('Device: {}'.format(device))

# Init model
model = EFT(model='l1')

def FLOPs_and_Patams(model, hw):
    from thop import profile
    dummy_input = torch.randn(1, 3, hw, hw)
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter: %.2fM" % (total/1e6))
    exit()

# FLOPs_and_Patams(model, 224)

# Transform frames
def transform(img):
    # print(img.shape)
    # img=cv2.resize(img,(500,500))
    transf = transforms.ToTensor()
    img_tensor = transf(img)  # (cxhxw)
    return img_tensor.unsqueeze(0)  # (bxcxhxw)

# Load pretrained model
c1 = 'checkpoints/MyNet_17-Sep_18-40-nodebs8-tep10-lr0.000357-wd0.1-56a1755a-892d-4e71-bdf2-258acfd41e5f_best.pt'
c2 = 'checkpoints/MyNet_19-Sep_16-45-nodebs10-tep25-lr0.000357-wd0.1-d99d94d7-bbf5-4ac9-a5ef-209eae0d4325_best.pt'
c3 = 'checkpoints/MyNet_19-Sep_21-00-nodebs10-tep25-lr0.000357-wd0.1-5979b5a3-c5a9-4558-a8af-1d91700b54fe_best.pt'
# model.load_state_dict(torch.load(c3), strict=False)
model.to(device)
model.eval()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()

    start = time.time()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply input transforms
    input_batch = transform(img).to(device)
    
    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = model(input_batch)
        
        # Delete last dim if it exist
        prediction = prediction.squeeze(0)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
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

    if SHOW_WINDOW:
        cv2.putText(img, f'FPS: %.2f' % fps, (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 2)
        cv2.imshow('Image', img)
        cv2.imshow('Depth Map', depth_map)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    # elif not os.path.exists('output'):
    #     os.mkdir('output')
    #     output_folder = 'output'
    #     file_name = 'output.jpg'
    # path = os.path.join(output_folder, file_name)
    # cv2.imwrite(path, depth_map)

cap.release()
