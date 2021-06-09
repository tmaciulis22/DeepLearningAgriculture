import numpy as np
import cv2
from PIL import Image
from matplotlib import image
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
from dataset import Dataset

model = torch.load('path/to/model')
CLASSES = ['weed_cluster','water','nutrient_deficiency']

test_dataset_vis = Dataset(
    "path/to/images", 
    "path/to/labels"
)
DEVICE = 'cuda'

for i in range(50):
    n = np.random.choice(len(test_dataset_vis))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset_vis[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    plt.figure()
    f, axarr = plt.subplots(2,3)


    axarr[0][0].imshow(image.transpose(1,2,0).astype('uint8'), cmap='gray')
    axarr[0][1].imshow(image.transpose(1,2,0).astype('uint8'), cmap='gray')
    axarr[0][2].imshow(image.transpose(1,2,0).astype('uint8'), cmap='gray')
    axarr[1][0].imshow(image.transpose(1,2,0).astype('uint8'), cmap='gray')
    axarr[1][1].imshow(image.transpose(1,2,0).astype('uint8'), cmap='gray')
    axarr[1][2].imshow(image.transpose(1,2,0).astype('uint8'), cmap='gray')
    axarr[0][0].imshow(gt_mask[0], cmap='jet', alpha=0.5)
    axarr[0][1].imshow(gt_mask[1], cmap='jet', alpha=0.5)
    axarr[0][2].imshow(gt_mask[2], cmap='jet', alpha=0.5)
    axarr[1][0].imshow(pr_mask[0], cmap='jet', alpha=0.5)
    axarr[1][1].imshow(pr_mask[1], cmap='jet', alpha=0.5)
    axarr[1][2].imshow(pr_mask[2], cmap='jet', alpha=0.5)
    plt.show()