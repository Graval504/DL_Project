import torch.optim as optim
from torch.utils.data import DataLoader
from timm.models.vision_transformer import VisionTransformer
from data_processing import open_data, open_PlantVillage, TreeSubset
from train import train, kfold, save_model
from torchmetrics import Accuracy
import torchvision.transforms as T
import torch.nn as nn

def finetune_plantVillage(model:VisionTransformer,train_transform:T.Compose = None, val_transform:T.Compose = None, batch_size:int=10, epochs=10, learning_rate=1e-5):
    dataset = open_PlantVillage(224,224)
    total_size = len(dataset)
    fraction = 1/5
    seg = int(total_size * fraction)
    trll = 0
    trlr = 0 * seg
    vall = trlr
    valr = 0 * seg + seg
    trrl = valr
    trrr = total_size
    train_left_indices = list(range(trll,trlr))
    train_right_indices = list(range(trrl,trrr))
    train_indices = train_left_indices + train_right_indices
    val_indices = list(range(vall,valr))

    train_set = TreeSubset(dataset,train_indices,train_transform)
    val_set = TreeSubset(dataset,val_indices,val_transform)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    metric = Accuracy(task='multiclass', num_classes=15)
    loss = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs * len(train_loader))
    val_loader = DataLoader(val_set, batch_size=batch_size,
                                        shuffle=True, num_workers=1)
    model.head = nn.Linear(model.head.in_features, 15)
    train_summary, val_summary = train(model=model, epochs=epochs, train_loader=train_loader, val_loader=val_loader,
                                        metric=metric, loss=loss, optimizer=optimizer, scheduler=scheduler)
    checkpoint_path = f'checkpoint/{type(model).__name__}_PlantVillage.pt'
    save_model(checkpoint_path, model, optimizer, scheduler, epochs+1)
    return model, train_summary, val_summary

def finetune(model:VisionTransformer, train_transform:T.Compose = None, val_transform:T.Compose = None, batch_size:int=10, epochs=10, learning_rate=1e-5):
    model.head = nn.Linear(model.head.in_features, 1)
    train_data = open_data("train", 224,224)
    test_data = open_data("test", 224,224)
    tr_summary, val_summary, test_summary = kfold(model,train_data,test_data,train_transform,val_transform,5,batch_size,epochs,learning_rate)
    return tr_summary, val_summary, test_summary