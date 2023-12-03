import torchvision.transforms as T
from data_processing import open_data, XLEN, YLEN
from train import kfold
from model import ResNet
import logging

def main():
    print(XLEN, YLEN)
    model = ResNet(3,[3, 3, 9, 3],[64, 128, 256, 512],1)    
    logging.basicConfig(
        filename=f'{type(model).__name__}.log',
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
    )

    train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((XLEN,YLEN),scale = (0.875,1)),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor()
        ])
    val_transform = T.Compose([
            T.ToTensor()
    ])

    train_data = open_data("train",train_transform)
    test_data = open_data("test", val_transform)
    train_summary, val_summary, test_summmary = kfold(model, train_data, test_data, 5, 5, 100)

    return




if __name__=="__main__":
    main()