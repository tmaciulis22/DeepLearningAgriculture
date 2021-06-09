from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import cv2
from matplotlib import pyplot as plt

import os

class Dataset(BaseDataset):
    CLASSES = ['weed_cluster','water','nutrient_deficiency']
    
    def __init__(
            self, 
            images_dir, 
            labels_dir
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.labels_dir = labels_dir
    
    def __getitem__(self, i):

        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = []
        for x in self.CLASSES:
            path = os.path.join(self.labels_dir, x ,self.ids[i].replace('.jpg','.png'))
            if(os.path.exists(path)):
                mask = cv2.imread(path, 0)/255
                masks.append(mask)
            else:
                masks.append(np.zeros((512,512)))
        return image.transpose(2, 0, 1).astype('float32'), np.stack(masks).astype('float32')
        
    def __len__(self):
        return len(self.ids)