import time
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from models.EFT import MyDepthModel


input_path = 'input'
output_path = 'output'
img_list = os.listdir(input_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device('cpu')  # Force use CPU
print('Device: {}'.format(device))

# Define model
model = MyDepthModel()

# Load pretrained model
c1 = 'checkpoints/MyNet_17-Sep_18-40-nodebs8-tep10-lr0.000357-wd0.1-56a1755a-892d-4e71-bdf2-258acfd41e5f_best.pt'
c2 = 'checkpoints/MyNet_19-Sep_16-45-nodebs10-tep25-lr0.000357-wd0.1-d99d94d7-bbf5-4ac9-a5ef-209eae0d4325_best.pt'
c3 = 'checkpoints/MyNet_19-Sep_21-00-nodebs10-tep25-lr0.000357-wd0.1-5979b5a3-c5a9-4558-a8af-1d91700b54fe_best.pt'
model.load_state_dict(torch.load(c3), strict=False)
model.to(device)
model.eval()

# Transform images
def transform(img):
    # img = cv2.resize(img,(500,500))
    transf = transforms.ToTensor()
    img_tensor = transf(img)  # (cxhxw)
    return img_tensor.unsqueeze(0)  # (bxcxhxw)

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

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_name = os.path.join(output_path, img_name)
    cv2.imwrite(output_name, depth_map)

print("All Done.")