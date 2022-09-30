import time
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from models import EFT


input_path = 'input'
output_path = 'output'
img_list = os.listdir(input_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cpu')  # Force use CPU
print('Device: {}'.format(device))

# Define model
model = EFT(model='l3')

# Load pretrained model
ckpt = ''
model.load_state_dict(torch.load(ckpt), strict=False)
model.to(device)
model.eval()

# Transform images
def transform(img):
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 544), interpolation=2)
    ])
    img = transf(img) 
    return img.unsqueeze(0)  # (bxcxhxw)

for img_name in img_list:
    img = os.path.join(input_path, img_name)
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    input = transform(img).to(device)
    start = time.time()

    # Prediction and resize to original resolution
    with torch.no_grad():
        pred = model(input)

        # Delete last dim if it exist
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
    print("Process " + img_name + " total Time: %.2f s" % totalTime)

    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)  # Color mode
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_name = os.path.join(output_path, img_name)
    cv2.imwrite(output_name, depth_map)

print("All Done.")