import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os




class Cat_Dog_Dataset(Dataset):
    def __init__(self, root, phase, data_aug=False):
        
        data_root = root

        self.phase = phase

        transforms1 = transforms.RandomRotation(degrees=30)
        transforms2 = transforms.RandomHorizontalFlip(p=0.5)
        self.transform = transforms.Compose([
                transforms.RandomApply([transforms1, transforms2], p=0.3)
            ])
        
        self.data_aug = data_aug

        if self.phase == 'train':
            train_cat = np.load(os.path.join(data_root, 'train_cat.npy'))
            train_dog = np.load(os.path.join(data_root, 'train_dog.npy'))
            cat_train_labels = np.zeros(len(train_cat), dtype=int)
            dog_train_labels = np.ones(len(train_dog), dtype=int)
            self.train_data = np.concatenate((train_cat, train_dog), axis=0)
            self.train_labels = np.concatenate((cat_train_labels, dog_train_labels), axis=0)
            train_cat, train_dog = 0, 0
            cat_train_labels, dog_train_labels = 0, 0
            train_permutation = np.random.permutation(len(self.train_data))
            self.train_data = torch.from_numpy(self.train_data[train_permutation])
            self.train_labels = torch.from_numpy(self.train_labels[train_permutation])
        elif self.phase == 'val':
            val_cat = np.load(os.path.join(data_root, 'val_cat.npy'))
            val_dog = np.load(os.path.join(data_root, 'val_dog.npy'))
            cat_val_labels = np.zeros(len(val_cat), dtype=int)
            dog_val_labels = np.ones(len(val_dog), dtype=int)
            self.val_data = np.concatenate((val_cat, val_dog), axis=0)
            self.val_labels = np.concatenate((cat_val_labels, dog_val_labels), axis=0)
            val_dog, val_cat = 0, 0
            cat_val_labels, dog_val_labels = 0, 0
            val_permutation = np.random.permutation(len(self.val_data))
            self.val_data = torch.from_numpy(self.val_data[val_permutation])
            self.val_labels = torch.from_numpy(self.val_labels[val_permutation])

    def __len__(self):
        if self.phase == 'train':
            return len(self.train_data)
        elif self.phase == 'val':
            return len(self.val_data)

    def __getitem__(self, idx):
        if self.phase == 'train':
            img = self.train_data[idx]
            label = self.train_labels[idx]
        elif self.phase == 'val':
            img = self.val_data[idx]
            label = self.val_labels[idx]
        
        if self.data_aug:
            img = self.transform(img)

        return img, label

