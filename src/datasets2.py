import os
import glob
import torch
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from .patch_extractor import PatchExtractor

LABELS = ['Normal', 'Benign', 'InSitu', 'Invasive']
#IMAGE_SIZE = (2048, 1536)
PATCH_SIZE = 224


class PatchWiseDataset2(Dataset):
    def __init__(self, path, stride=PATCH_SIZE, rotate=False, flip=False, enhance=False):
        super().__init__()

        wp = int((448 - PATCH_SIZE) / stride + 1)
        hp = int((336 - PATCH_SIZE) / stride + 1)
        labels = {name: index for index in range(len(LABELS)) for name in glob.glob(path + '/' + LABELS[index] + '/*.tif')}

        self.path = path
        self.stride = stride
        self.labels = labels
        self.names = list(sorted(labels.keys()))
        self.shape = (len(labels), wp, hp, (4 if rotate else 1), (2 if flip else 1), (2 if enhance else 1))  # (files, x_patches, y_patches, rotations, flip, enhance)
        self.augment_size = np.prod(self.shape) / len(labels)

    def __getitem__(self, index):
        im, xpatch, ypatch, rotation, flip, enhance = np.unravel_index(index, self.shape)

        with Image.open(self.names[im]) as img:
            
            img1 = img.resize((448, 336)) #scale II
            extractor = PatchExtractor(img=img1, patch_size=PATCH_SIZE, stride=self.stride)
            patch = extractor.extract_patch((xpatch, ypatch))

            if rotation != 0:
                patch = patch.rotate(rotation * 90)

            if flip != 0:
                patch = patch.transpose(Image.FLIP_LEFT_RIGHT)

            if enhance != 0:
                factors = np.random.uniform(.5, 1.5, 3)
                patch = ImageEnhance.Color(patch).enhance(factors[0])
                patch = ImageEnhance.Contrast(patch).enhance(factors[1])
                patch = ImageEnhance.Brightness(patch).enhance(factors[2])

            label = self.labels[self.names[im]]
            return transforms.ToTensor()(patch), label

    def __len__(self):
        return np.prod(self.shape)


class ImageWiseDataset2(Dataset):
    def __init__(self, path, stride=PATCH_SIZE, rotate=False, flip=False, enhance=False):
        super().__init__()

        labels = {name: index for index in range(len(LABELS)) for name in glob.glob(path + '/' + LABELS[index] + '/*.tif')}

        self.path = path
        self.stride = stride
        self.labels = labels
        self.names = list(sorted(labels.keys()))
        self.shape = (len(labels), (2 if rotate else 1), (2 if flip else 1), (2 if enhance else 1))  # (files, x_patches, y_patches, rotations, flip, enhance)
        self.augment_size = np.prod(self.shape) / len(labels)

    def __getitem__(self, index):
        im, rotation, flip, enhance = np.unravel_index(index, self.shape)

        with Image.open(self.names[im]) as img:
            
            img1 = img.resize((448, 336)) #scale II
            if flip != 0:
                img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            

            if rotation != 0:
                img1 = img1.rotate(rotation * 180)

        
            if enhance != 0:
                factors = np.random.uniform(.5, 1.5, 3)
                img1 = ImageEnhance.Color(img1).enhance(factors[0])
                img1 = ImageEnhance.Contrast(img1).enhance(factors[1])
                img1 = ImageEnhance.Brightness(img1).enhance(factors[2])

            extractor = PatchExtractor(img=img1, patch_size=PATCH_SIZE, stride=self.stride)
            patches = extractor.extract_patches()

            label = self.labels[self.names[im]]

            b = torch.zeros((len(patches), 3, PATCH_SIZE, PATCH_SIZE))
            for i in range(len(patches)):
                b[i] = transforms.ToTensor()(patches[i])

            return b, label

    def __len__(self):
        return np.prod(self.shape)


class TestDataset2(Dataset):
    def __init__(self, path, stride=PATCH_SIZE, augment=False):
        super().__init__()

        if os.path.isdir(path):
            names = [name for name in glob.glob(path + '/*.tif')]
        else:
            names = [path]

        self.path = path
        self.stride = stride
        self.augment = augment
        self.names = list(sorted(names))

    def __getitem__(self, index):
        file = self.names[index]
        with Image.open(file) as img:
            img1 = img.resize((448, 336)) #scale II
            bins = 8 if self.augment else 1
            extractor = PatchExtractor(img=img1, patch_size=PATCH_SIZE, stride=self.stride)
            b = torch.zeros((bins, extractor.shape()[0] * extractor.shape()[1], 3, PATCH_SIZE, PATCH_SIZE))

            for k in range(bins):

                if k % 4 != 0:
                    img1 = img1.rotate((k % 4) * 90)

                if k // 4 != 0:
                    img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)

                extractor = PatchExtractor(img=img1, patch_size=PATCH_SIZE, stride=self.stride)
                patches = extractor.extract_patches()

                for i in range(len(patches)):
                    b[k, i] = transforms.ToTensor()(patches[i])

            return b, file

    def __len__(self):
        return len(self.names)
