import os
from pathlib import Path

import cv2
import torch.utils.data as data
import numpy as np


class TestDataSet(data.Dataset):
    def __init__(self, data_path, opt):
        self.data_path = data_path

        # load annotations that meet specific standards
        self.img_dir = Path(data_path, split + str(self.coco_year))
        self.fnames = None
        self.opt = opt
        print('Loaded {} images for {}'.format(len(self.indices), split))

    def get_item_raw(self, idx):
        fname = self.fnames[idx]
        img_path = Path(self.img_dir, fname)
        img = self.load_image(img_path)
        return img

    def __getitem__(self, index):
        img = self.get_item_raw(index)
        img = normalize(img)
        return img

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = img.astype('float32') / 255.
        return img

    def __len__(self):
        return len(self.indices)
