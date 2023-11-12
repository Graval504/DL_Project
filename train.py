import logging
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.aggregation import MeanMetric
from torchmetrics import Accuracy
from data_processing import load_data

def train(model, epochs):
    train_loader, val_loader = load_data(batch_size=128)
    optimizer = optim.AdamW(model.parameters(), lr=1e3, betas=(0.9, 0.999))
    metric = Accuracy(task='binary', num_classes=2)
    loss = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader))

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
        checkpoint_path = f'checkpoint/{model}-{optimizer}with{loss}_last.pt'
        save_model(checkpoint_path, model, optimizer, scheduler, epoch+1)

def train_one_epoch(model, loader, metric_fn, loss_fn, device, optimizer, scheduler):
    # set model to train mode    
    model.train()

    # average meters to trace loss and accuracy
    loss_epoch = MeanMetric()
    accuracy_epoch = MeanMetric()    
    # train loop
    for inputs, targets in tqdm.tqdm(loader):
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