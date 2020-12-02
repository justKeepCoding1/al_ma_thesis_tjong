import os
from os.path import dirname as dr, abspath
from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets


# MNIST that allows index access and limiting dataset size
# wrap/inherit original MNIST dataset from pytorch
# shuffle = False is recommended to allow data searching


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

class Dataset_F_MNIST_n(datasets.FashionMNIST):

    def __init__(self, root, train:bool=True, transform=None, target_transform=None,
                 download=True, n=10000):
        '''

        :param root: location of directory names with this class name
        :param train: True to define training datasset, False for test dataset
        :param transform: Transformation on MNIST data, usually normalization
        :param target_transform: transformation to target/GT
        :param download: Flag to allow dataset download in case it is not available
        :param n: Number of MNIST data loaded
        '''
        super(datasets.FashionMNIST, self).__init__(root, transform=transform, download=download,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        self.n = n

    def __getitem__(self, index):

        img, target = super(Dataset_F_MNIST_n, self).__getitem__(index)

        return img, target, index

    def __len__(self):
        return self.n


# Dataset to access reduced MNIST
class Reduced_F_MNIST(Dataset):
    def __init__(self, root, train=True):
        '''

        :param root: location of coded_mnist_train.pt and coded_mnist_test.pt
        :param train: True to access training data, false for test data
        MNIST was normalized using transforms.Normalize((0.1307,), (0.3081,)) before reduced
        '''
        if train:
            path = os.path.join(root, "reduced_f_mnist_train.pt")
        else:
            path = os.path.join(root, "reduced_f_mnist_test.pt")

        self.data, self.target, self.data_index = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get true index as defined in MNIST, referring to data_index
        idx = self.data_index[tuple((self.data_index == idx).nonzero()[0].numpy())].item()

        return self.data[idx], self.target[idx], idx


# MNIST that allows index access and limiting dataset size
# wrap/inherit original MNIST dataset from pytorch
class Dataset_MNIST_n(datasets.MNIST):

    def __init__(self, root, train:bool=True, transform=None, target_transform=None,
                 download=False, n=10000):
        '''

        :param root: location of directory names with this class name
        :param train: True to define training datasset, False for test dataset
        :param transform: Transformation on MNIST data, usually normalization
        :param target_transform: transformation to target/GT
        :param download: Flag to allow dataset download in case it is not available
        :param n: Number of MNIST data loaded
        '''
        # super(Dataset_Cityscapes_n, self) calls the parent class instance of current class instance
        super(datasets.MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        self.n = n

    def __getitem__(self, index):

        img, target = super(Dataset_MNIST_n, self).__getitem__(index)

        return img, target, index

    def __len__(self):
        return self.n


# Dataset to access reduced MNIST
class Reduced_MNIST(Dataset):
    def __init__(self, root, train=True):
        '''

        :param root: location of coded_mnist_train.pt and coded_mnist_test.pt
        :param train: True to access training data, false for test data
        MNIST was normalized using transforms.Normalize((0.1307,), (0.3081,)) before reduced
        '''
        if train:
            path = os.path.join(root, "reduced_mnist_train.pt")
        else:
            path = os.path.join(root, "reduced_mnist_test.pt")

        self.data, self.target, self.data_index = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get true index as defined in MNIST, referring to data_index
        idx = self.data_index[tuple((self.data_index == idx).nonzero()[0].numpy())].item()

        return self.data[idx], self.target[idx], idx


def provide_reduced_mnist(train=True):

    mnist_root = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 'Dataset_MNIST_n')
    if train:
        # dataset
        dataset = Reduced_MNIST(root=mnist_root, train=True)
        loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)

    else:
        dataset = Reduced_MNIST(root=mnist_root, train=False)
        loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)

    data_torch, target_torch, _ = next(iter(loader))
    data, target = data_torch.numpy(), target_torch.numpy()

    return data, target


def provide_unreduced_mnist(train=True):
    mnist_root = os.path.join(dr(dr(dr(abspath(__file__)))), 'data',)
    transform = transforms.Compose([transforms.ToTensor()])

    if train:
        # dataset
        dataset = Dataset_MNIST_n(root=mnist_root, train=True, n=60000, transform=transform)
        loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)

    else:
        dataset = Dataset_MNIST_n(root=mnist_root, train=False, n=10000, transform=transform)
        loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)

    data_torch, target_torch, _ = next(iter(loader))
    data, target = data_torch.numpy().squeeze(axis=1).reshape(dataset.__len__(), 28 * 28), target_torch.numpy()

    return data, target


def provide_reduced_f_mnist(train=True):

    mnist_root = os.path.join(dr(dr(dr(abspath(__file__)))), 'data', 'Dataset_F_MNIST_n')

    if train:
        # dataset
        dataset = Reduced_F_MNIST(root=mnist_root, train=True)
        loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)

    else:
        dataset = Reduced_F_MNIST(root=mnist_root, train=False)
        loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)

    data_torch, target_torch, _ = next(iter(loader))
    data, target = data_torch.numpy(), target_torch.numpy()

    return data, target


def provide_unreduced_f_mnist(train=True):
    mnist_root = os.path.join(dr(dr(dr(abspath(__file__)))), 'data')
    transform = transforms.Compose([transforms.ToTensor()])

    if train:
        # dataset
        dataset = Dataset_F_MNIST_n(root=mnist_root, train=True, n=60000, transform=transform)
        loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)

    else:
        dataset = Dataset_F_MNIST_n(root=mnist_root, train=False, n=10000, transform=transform)
        loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)

    data_torch, target_torch, _ = next(iter(loader))
    data, target = data_torch.numpy().squeeze(axis=1).reshape(dataset.__len__(), 28 * 28), target_torch.numpy()

    return data, target


'''
# Custom dataset to work with MNIST with segmentation
class MNIST_JPG_Dataset(Dataset):
    def __init__(self, root, train_flag, n_data, transform=None):
        

        :param root: location of train and test folder
        :param train_flag: True tor train data and false for test data
        :param n_data: number of data loaded
        :param transform: trandformation on dataset
        
        super().__init__()
        if root[-1] == "/":
            root = root[:-1]
        self.root = root
        self.train_flag = train_flag
        self.n_data = n_data
        self.transform = transform

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        if self.train_flag:
            jpg_root = self.root + "/train/img/img" + str(index) + ".jpg"
            seg_root = self.root + "/train/gt/img" + str(index) + ".png"
        else:
            jpg_root = self.root + "/test/img/img" + str(index) + ".jpg"
            seg_root = self.root + "/test/gt/img" + str(index) + ".png"

        img = transforms.ToTensor()(Image.open(jpg_root)).type(torch.float32)
        gt = transforms.ToTensor()(Image.open(seg_root))

        # process segmentation gt
        gt = np.swapaxes(gt.numpy(), 0, 2)
        gt_label = np.zeros((gt.shape[0], gt.shape[1]))
        for i_color in range(label_colors.__len__()):
            gt_temp = np.where(np.all(np.round_(gt, decimals=4) == np.round_(np.array(label_colors[i_color])/255, decimals=4).astype('float32'), axis=2), i_color, 0)
            gt_label += gt_temp
        gt_label = torch.from_numpy(gt_label).type(torch.long)

        # transform images

        return img, gt_label'''