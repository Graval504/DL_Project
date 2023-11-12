import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Literal
import glob
from PIL.Image import open
import PIL.Image
import torch

PIL.Image.MAX_IMAGE_PIXELS = None

def load_data(batch_size:int)->tuple[DataLoader,DataLoader]:
    train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((32,32),scale = (0.875,1)),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
        ])
    train_data = open_data("train")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_transform = T.Compose([
        T.ToTensor(),
    ])
    val_data = open_data("test")
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader,val_loader

def open_data(dir:Literal["train","test"]):
    '''
    dir : "train" or "test"
    '''
    healthy_list = [T.ToTensor()(open(data)) for data in glob.glob(f"./data/{dir}/healthy/*.jpg")]
    disease_list = [T.ToTensor()(open(data)) for data in glob.glob(f"./data/{dir}/disease/*.jpg")]
    
    data_list = healthy_list + disease_list
    label_list = torch.cat([torch.zeros(len(healthy_list)),torch.ones(len(disease_list))])

    return TreeDataset(data_list,label_list)

class TreeDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return (data, label)