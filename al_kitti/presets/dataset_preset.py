import os

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets


class Dataset_Cityscapes_n(datasets.Cityscapes):

    # init copied from parent class Cityscapes
    def __init__(self, root, split='train', mode='fine', target_type='color',
                 transform=None, target_transform=None, transforms=None, n=1000):
        # super(Dataset_Cityscapes_n, self) calls the parent class instance of current class instance
        super(Dataset_Cityscapes_n, self).__init__(root, transform=transform, split=split, mode=mode,
                                                   transforms=transforms,
                                                   target_type=target_type,
                                                   target_transform=target_transform)

        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []
        self.n = n

        if not isinstance(target_type, list):
            self.target_type = [target_type]

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

    def __getitem__(self, index):

        img, target = super(Dataset_Cityscapes_n, self).__getitem__(index)

        return img, target

    def __getitem_index__(self, index):

        img, target = super(Dataset_Cityscapes_n, self).__getitem__(index)

        return img, target, index

    def __len__(self):
        return self.n


class Dataset_Cityscapes_n_i(datasets.Cityscapes):

    # init copied from parent class Cityscapes
    def __init__(self, root, split='train', mode='fine', target_type='color',
                 transform=None, target_transform=None, transforms=None, n=1000):
        # super(Dataset_Cityscapes_n, self) calls the parent class instance of current class instance
        super(Dataset_Cityscapes_n_i, self).__init__(root, transform=transform, split=split, mode=mode,
                                                     transforms=transforms, target_type=target_type,
                                                     target_transform=target_transform)

        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []
        self.n = n

        if not isinstance(target_type, list):
            self.target_type = [target_type]

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

    def __getitem__(self, index):

        img, target = super(Dataset_Cityscapes_n_i, self).__getitem__(index)
        img_name = os.path.splitext(os.path.basename(self.images[index]))[0]

        return img, target, index, img_name

    def __len__(self):
        return self.n

