import logging
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.aggregation import MeanMetric
from torchmetrics import Accuracy
import torchvision.transforms as T
from data_processing import TreeDataset, TreeSubset, TreeDatasetWithTransform
import numpy as np
from model import VotingClassifier
import pandas as pd
import copy

def train(model, epochs, train_loader, val_loader, metric, loss, optimizer, scheduler):

    for epoch in range(epochs):
        # train one epoch
        train_summary = train_one_epoch(
            model, train_loader, metric, loss, 'cuda',
            optimizer, scheduler)

        log = (f'epoch {epoch+1}, '
                + f'train_loss: {train_summary["loss"]:.4f}, '
                + f'train_accuracy: {train_summary["accuracy"]:.4f}')
        print(log)
        logging.info(log)
        
        # evaluate one epoch
        val_summary = eval_one_epoch(
            model, val_loader, metric, loss, 'cuda'
        )
        log = (f'epoch {epoch+1}, '
                + f'val_loss: {val_summary["loss"]:.4f}, '
                + f'val_accuracy: {val_summary["accuracy"]:.4f}')
        print(log)
        logging.info(log)

        # save model
        checkpoint_path = f'checkpoint/{type(model).__name__}_last.pt'
        save_model(checkpoint_path, model, optimizer, scheduler, epoch+1)

    return train_summary, val_summary

def eval(model, test_loader, metric, loss):
    test_summary = eval_one_epoch(
        model, test_loader, metric, loss, 'cuda'
    )
    log = (f'test_loss: {test_summary["loss"]:.4f}, '
           + f'test_accuracy: {test_summary["accuracy"]:.4f}')
    print(log)
    logging.info(log)
    return test_summary

def train_one_epoch(model, loader, metric_fn, loss_fn, device, optimizer, scheduler):
    # set model to train mode    
    model.train()

    # average meters to trace loss and accuracy
    loss_epoch = MeanMetric()
    accuracy_epoch = MeanMetric()    
    # train loop
    for inputs, targets in tqdm.tqdm(loader):
        if type(loss_fn) == nn.BCEWithLogitsLoss and len(targets.shape) == 1:
            targets = targets.unsqueeze(1)
        # move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        # forward
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        accuracy = metric_fn(outputs, targets)
        
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # update statistics
        loss_epoch.update(loss.to('cpu'))
        accuracy_epoch.update(accuracy.to('cpu'))
    
    summary = {
        'loss': loss_epoch.compute(),
        'accuracy': accuracy_epoch.compute(),
    }

    return summary


def eval_one_epoch(model, loader, metric_fn, loss_fn, device):
    # set model to evaluatinon mode    
    model.eval()
    
    # average meters to trace loss and accuracy
    loss_epoch = MeanMetric()
    accuracy_epoch = MeanMetric()    
    # train loop
    for inputs, targets in tqdm.tqdm(loader):
        if type(loss_fn) == nn.BCEWithLogitsLoss and len(targets.shape) == 1:
            targets = targets.unsqueeze(1)
        # move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # forward
        with torch.no_grad():
            outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        accuracy = metric_fn(outputs, targets)
        
        # update statistics
        loss_epoch.update(loss.to('cpu'))
        accuracy_epoch.update(accuracy.to('cpu'))
    
    summary = {
        'loss': loss_epoch.compute(),
        'accuracy': accuracy_epoch.compute(),
    }

    return summary


def save_model(path, model, optimizer, scheduler, epoch):
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state_dict, path)

def kfold(base_model:nn.Module=None, train_dataset:TreeDataset=None, test_dataset:TreeDataset= None, train_transform:T.Compose = None, val_transform:T.Compose = None, k_fold=5, batch_size:int=10, epochs=10, learning_rate=1e-3):
    train_summaries = pd.Series()
    val_summaries = pd.Series()
    total_size = len(train_dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    folds = []
    
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        model = copy.deepcopy(base_model)
        model = model.to("cuda")
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        metric = Accuracy(task='binary', num_classes=1)
        loss = nn.BCEWithLogitsLoss()
        metric = metric.to("cuda")
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        # msg
#         print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" 
#               % (trll,trlr,trrl,trrr,vall,valr))
        
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        train_set = TreeSubset(train_dataset,train_indices,train_transform)
        val_set = TreeSubset(train_dataset,val_indices,val_transform)
#         print(len(train_set),len(val_set))
#         print()
        
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader))
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
        train_summary, val_summary = train(model=model, epochs=epochs, train_loader=train_loader, val_loader=val_loader,
                                           metric=metric, loss=loss, optimizer=optimizer, scheduler=scheduler)
        train_summaries.at[i] = train_summary
        val_summaries.at[i] = val_summary
        log = (f'{i+1} fold, '
                + f'loss: {val_summary["loss"]:.4f}, '
                + f'accuracy: {val_summary["accuracy"]:.4f}')
        checkpoint_path = f'checkpoint/{type(model).__name__}_{i+1}fold.pt'
        save_model(checkpoint_path, model, optimizer, scheduler, epochs+1)
        folds.append(model)
        print(log)
        logging.info(log)
    ensemble = VotingClassifier(folds)
    test_summary = evaluate_test(ensemble, test_dataset, metric, loss, batch_size)
    return train_summaries, val_summaries, test_summary

def evaluate_test(model, test_dataset:TreeDataset, metric, loss, batch_size=5):
    test_transform = T.Compose([T.ToTensor()])
    test_dataset = TreeDatasetWithTransform(test_dataset, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)
    test_summary, conf = eval_with_confusion_matrix(model, test_loader, metric_fn=metric, loss_fn=loss, device="cuda")
    log = (f'test set evaluate, '
                + f'loss: {test_summary["loss"]:.4f}, '
                + f'accuracy: {test_summary["accuracy"]:.4f}')
    print(log)
    logging.info(log)
    return test_summary, conf
        
def eval_with_confusion_matrix(model, loader, metric_fn, loss_fn, device):
    # set model to evaluatinon mode    
    model.eval()
    
    # average meters to trace loss and accuracy
    loss_epoch = MeanMetric()
    accuracy_epoch = MeanMetric()    
    output_list = []
    target_list = []
    # loop
    for inputs, targets in tqdm.tqdm(loader):
        if type(loss_fn) == nn.BCEWithLogitsLoss and len(targets.shape) == 1:
            targets = targets.unsqueeze(1)
        # move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # forward
        with torch.no_grad():
            outputs = model(inputs)
        output_list.append(outputs)
        target_list.append(targets)
        loss = loss_fn(outputs, targets)
        accuracy = metric_fn(outputs, targets)
        
        # update statistics
        loss_epoch.update(loss.to('cpu'))
        accuracy_epoch.update(accuracy.to('cpu'))
    output_list = torch.flatten(torch.cat(output_list, dim=0))
    target_list = torch.flatten(torch.cat(target_list, dim=0))
    output_list = output_list.to("cpu")
    target_list = target_list.to("cpu")
    output_list = (torch.sigmoid(output_list)>0.5).int()
    target_list = target_list.int()
    tp, tn, fp, fn = 0, 0, 0, 0
    for o, t in zip(output_list,target_list):
        if o==t:
            if o == 0:
                tn+=1
                continue
            tp+=1
            continue
        if o == 0:
            fn+=1
            continue
        fp +=1
    conf = np.array([[tn,fp],[fn,tp]])
    summary = {
        'loss': loss_epoch.compute(),
        'accuracy': accuracy_epoch.compute(),
    }
    return summary, conf