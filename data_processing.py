import torchvision.transforms as T
from torchvision.transforms.functional import pad
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from torch import Tensor
from typing import Literal, Union, Sequence, List
import glob
from PIL.Image import open, Image, Transpose, Resampling
import PIL.Image
import torch

PIL.Image.MAX_IMAGE_PIXELS = None

XLEN = 584 #1168
YLEN = 335 #669

def load_data(batch_size:int)->tuple[DataLoader,DataLoader]:
    train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((XLEN, YLEN),scale = (0.875,1)),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor()
        ])
    train_data = open_data("train",train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True,
                              num_workers=1)
    val_transform = T.Compose([
            T.ToTensor()
    ])
    val_data = open_data("test", val_transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader

class TreeDataset(Dataset):
    def __init__(self, data:list[Image], label:Tensor):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx) -> tuple[Image, Tensor]:
        data = self.data[idx]
        label = self.label[idx]
        return data, label
    
def open_data(dir:Literal["train","test"]) -> TreeDataset:
    '''
    dir : "train" or "test"
    '''
    healthy_list = [image_resize(open(data)) for data in glob.glob(f"./data/{dir}/healthy/*.jpg")]
    disease_list = [image_resize(open(data)) for data in glob.glob(f"./data/{dir}/disease/*.jpg")]
    
    data_list = healthy_list + disease_list
    label_list = torch.cat([torch.zeros(len(healthy_list)),torch.ones(len(disease_list))])
    return TreeDataset(data_list,label_list)

def tree_collate_fn(samples:TreeDataset):
    collate_X = []
    collate_y = []
    for data, label in samples:
        x, y = data.size
        if x < y:
            data = data.transpose(Transpose.ROTATE_90)
            buf = x
            x = y
            y = buf
        if XLEN*y < YLEN*x:
            y = XLEN*y//x
            diff = YLEN-y
            data = data.resize((XLEN,y),Resampling.LANCZOS)
            zero_pad1 = torch.zeros(size=(3, diff//2, XLEN))
            zero_pad2 = torch.zeros(size=(3, diff//2+diff%2, XLEN))
            catdim = 1
        else:
            x = YLEN*x//y
            diff = XLEN-x
            data = data.resize((x,YLEN),Resampling.LANCZOS)
            zero_pad1 = torch.zeros(size=(3, YLEN, diff//2))
            zero_pad2 = torch.zeros(size=(3, YLEN, diff//2+diff%2))
            catdim = 2
        data = T.ToTensor()(data)
        collate_X.append(torch.cat([zero_pad1, data, zero_pad2], dim=catdim))
    collate_y.append(label)
    return torch.stack(collate_X), torch.stack(collate_y)

def image_resize(data:Image):
    x, y = data.size
    if x < y:
        data = data.transpose(Transpose.ROTATE_90)
        buf = x
        x = y
        y = buf
    if XLEN*y < YLEN*x:
        y = XLEN*y//x
        diff = YLEN-y
        data = data.resize((XLEN,y),Resampling.LANCZOS)
        zero_pad1 = diff//2
        zero_pad2 = diff//2+diff%2
        data = pad(data,(0,zero_pad1,0,zero_pad2))
    else:
        x = YLEN*x//y
        diff = XLEN-x
        data = data.resize((x,YLEN),Resampling.LANCZOS)
        zero_pad1 = diff//2
        zero_pad2 = diff//2+diff%2
        data = pad(data,(zero_pad1,0,zero_pad2,0))
    return data

class TreeSubset(Subset):
    def __init__(self, dataset: TreeDataset, indices: Sequence[int], transform:T.Compose) -> None:
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        data, label = super().__getitem__(idx)
        return (self.transform(data),label)
    
    def __getitems__(self, indices: List[int]) -> List:
        res = [(self.transform(data),label) for data,label in super().__getitems__(indices)]
        return res

class TreeDatasetWithTransform(Dataset):
    def __init__(self, dataset: TreeDataset, transform: T.Compose):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx) -> tuple[Image, Tensor]:
        data, label = self.dataset.__getitem__(idx)
        return (self.transform(data), label)
    
    def __len__(self):
        return len(self.dataset.label)