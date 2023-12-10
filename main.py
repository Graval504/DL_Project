import torchvision.transforms as T
from data_processing import open_data, XLEN, YLEN
from train import kfold, evaluate_test
from model import ResNet, load_model, VotingClassifier
from torchmetrics import Accuracy
import torch.nn as nn
import torch.optim as optim
import logging

blocks=[2, 2, 2, 2] #[3, 3, 9, 3]
dims=[64, 128, 256, 512]

def main():
    print(XLEN, YLEN)
    model = ResNet(3,blocks,dims,1)    
    logging.basicConfig(
        filename=f'{type(model).__name__}.log',
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
    )

    train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((XLEN,YLEN),scale = (0.875,1)),
            #T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor()
        ])
    val_transform = T.Compose([
            T.ToTensor()
    ])

    train_data = open_data("train")
    test_data = open_data("test")
    train_summary, val_summary, test_summmary = kfold(model, train_data, test_data, train_transform, val_transform, 5, 5, 100)
    return

def load():
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


if __name__=="__main__":
    main()
    #load()