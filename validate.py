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

CLASSES = ['weed_cluster','water','nutrient_deficiency']

DEVICE = 'cuda'

if __name__ == '__main__':
    model = torch.load('path/to/model')

    valid_dataset = Dataset(
        "path/to/images", 
        "path/to/labels"
    )

    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(),
        smp.utils.metrics.Fscore(),
        smp.utils.metrics.Accuracy(),
        smp.utils.metrics.Recall(),
        smp.utils.metrics.Precision(),
    ]

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    valid_logs = valid_epoch.run(valid_loader)
    print(valid_logs)
