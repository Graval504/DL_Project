import torchvision.transforms as T
from data_processing import open_data
from train import kfold, evaluate_test
from model import ResNet, load_model, VotingClassifier
from finetune import finetune, finetune_plantVillage
from torchmetrics import Accuracy
import torch.nn as nn
import torch.optim as optim
import logging
import timm

blocks=[2, 2, 2, 2] #[3, 3, 9, 3]
dims=[64, 128, 256, 512]

def main_resnet():
    XLEN, YLEN = 584, 335
    model = ResNet(3,blocks,dims,1)    
    logging.basicConfig(
        filename=f'{type(model).__name__}.log',
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
    )

    train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((XLEN, YLEN),scale = (0.875,1)),
            #T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor()
        ])
    val_transform = T.Compose([
            T.ToTensor()
    ])

    train_data = open_data("train", XLEN, YLEN)
    test_data = open_data("test", XLEN, YLEN)
    train_summary, val_summary, test_summmary = kfold(model, train_data, test_data, train_transform, val_transform, 5, 5, 100, 5e-4)
    return

def main_loadmodel():
    logging.basicConfig(
        filename='ensemble.log',
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
    )
    model = ResNet(3,blocks,dims,1)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100 * 87)
    models = []
    for i in range(5):
        model, optimizer, scheduler = load_model(f"checkpoint/ResNet_{i+1}fold.pt",model, optimizer, scheduler)
        models.append(model.to("cuda"))
    ensemble = VotingClassifier(models)
    test_dataset = open_data("test")
    metric = Accuracy(task='binary', num_classes=1).to("cuda")
    loss = nn.BCEWithLogitsLoss()
    test_summary = evaluate_test(ensemble, test_dataset, metric, loss, 5)
    return

def main_finetune():
    train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((224, 224),scale = (0.875,1)),
            #T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor()
        ])
    val_transform = T.Compose([
            T.ToTensor()
    ])
    logging.basicConfig(
        filename=f'ViTS.log',
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
    )
    model = timm.create_model("vit_base_patch16_224.augreg_in21k",pretrained=True)
    train_summary, val_summary, test_summmary = finetune(model, train_transform, val_transform, 10, 10, 1e-4)
    return

def main_finetuneWithPlantVillage():
    train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((224, 224),scale = (0.875,1)),
            #T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor()
        ])
    val_transform = T.Compose([
            T.ToTensor()
    ])
    logging.basicConfig(
        filename=f'ViTS-PlantVillage.log',
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
    )
    model = timm.create_model("vit_base_patch16_224.augreg_in21k",pretrained=True)
    model, train_summary, val_summary = finetune_plantVillage(model,train_transform,val_transform,10,10,1e-4)
    train_summary, val_summary, test_summmary = finetune(model, train_transform, val_transform, 10, 10, 2e-5)
    return

if __name__=="__main__":
    #main_resnet()
    #main_loadmodel()
    main_finetune()
    #main_finetuneWithPlantVillage()