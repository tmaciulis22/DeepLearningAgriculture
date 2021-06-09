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

ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['weed_cluster','water','nutrient_deficiency']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

if __name__ == '__main__':
    model = smp.DeepLabV3(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        "path/to/images", 
        "path/to/labels"
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    for i in range(0, 20):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        torch.save(model, 'path_to_model_{}.pth'.format(i))