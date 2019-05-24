import os
from pathlib import Path

import cv2
import torch.utils.data as data
import numpy as np
import random
from data_process.process_utils import normalize


class TestDataSet(data.Dataset):
    def __init__(self, data_path, fnames, opt):
        self.data_path = data_path

        # load annotations that meet specific standards
        self.img_dir = data_path
        self.fnames = fnames
        self.opt = opt
        print('Loaded {} images'.format(len(self.fnames)))

    def get_item_raw(self, idx):
        fname = self.fnames[idx]
        img_path = Path(self.img_dir, fname)
        img = self.load_image(img_path)
        return img

    def __getitem__(self, index):
        img = self.get_item_raw(index)
        img = normalize(img)
        return np.array([img]).astype('float32')

    def load_image(self, img_path):
        img = cv2.imread(str(img_path))
        img = self.crop_image(img)
        img = img.astype('float32') / 255.
        return img

    def crop_image(self, img):
        h, w, _ = img.shape

        rnd_h_max = h - self.opt.imgSize - 1
        rnd_h_max = 0 if rnd_h_max<0 else rnd_h_max
        rnd_w_max = w - self.opt.imgSize - 1
        rnd_w_max = 0 if rnd_w_max<0 else rnd_w_max

        s_crop_h = random.randint(0, rnd_h_max)
        e_crop_h = s_crop_h + self.opt.imgSize
        s_crop_w = random.randint(0, rnd_w_max)
        e_crop_w = s_crop_w + self.opt.imgSize
        img = img[s_crop_h:e_crop_h, s_crop_w:e_crop_w, :]
        return img

    def __len__(self):
        return len(self.fnames)
