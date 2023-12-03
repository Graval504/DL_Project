import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from torch import Tensor
from typing import Literal, Union, Sequence
import glob
from PIL.Image import open, Image, Transpose, Resampling
import PIL.Image
import torch

PIL.Image.MAX_IMAGE_PIXELS = None

def load_data(batch_size:int)->tuple[DataLoader,DataLoader]:
    train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((1168,669),scale = (0.875,1)),
            T.RandAugment(num_ops=2, magnitude=9),
        ])
    train_data = open_data("train",train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=1, collate_fn= tree_collate_fn)
    val_transform = T.Compose([
    ])
    val_data = open_data("test", val_transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn= tree_collate_fn, shuffle=True)

    return train_loader,val_loader

class TreeDataset(Dataset):
    def __init__(self, data:list[Image], label:Tensor, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx) -> tuple[Image, Tensor]:
        data = self.transform(self.data[idx])
        label = self.label[idx]
        return data, label
    
    def change_transform(self, transform):
        self.transform = transform
    
def open_data(dir:Literal["train","test"], transform) -> TreeDataset:
    '''
    dir : "train" or "test"
    '''
    healthy_list = [open(data) for data in glob.glob(f"./data/{dir}/healthy/*.jpg")]
    disease_list = [open(data) for data in glob.glob(f"./data/{dir}/disease/*.jpg")]
    
    data_list = healthy_list + disease_list
    label_list = torch.cat([torch.zeros(len(healthy_list)),torch.ones(len(disease_list))])
    return TreeDataset(data_list,label_list,transform)

def tree_collate_fn(samples:TreeDataset):
    collate_X = []
    collate_y = []
    xlen = 1168
    ylen = 669
    for data, label in samples:
        x, y = data.size
        if x < y:
            data = data.transpose(Transpose.ROTATE_90)
            buf = x
            x = y
            y = buf
        if 1168*y < 669*x:
            y = xlen*y//x
            diff = ylen-y
            data = data.resize((xlen,y),Resampling.LANCZOS)
            zero_pad1 = torch.zeros(size=(3, diff//2, 1168))
            zero_pad2 = torch.zeros(size=(3, diff//2+diff%2, 1168))
            catdim = 1
        else:
            x = ylen*x//y
            diff = xlen-x
            data = data.resize((x,ylen),Resampling.LANCZOS)
            zero_pad1 = torch.zeros(size=(3, 669, diff//2))
            zero_pad2 = torch.zeros(size=(3, 669, diff//2+diff%2))
            catdim = 2
        data = T.ToTensor()(data)
        collate_X.append(torch.cat([zero_pad1, data, zero_pad2], dim=catdim))
    collate_y.append(label)
    return torch.stack(collate_X), torch.stack(collate_y)