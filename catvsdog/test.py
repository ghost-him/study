from model import LeNet
import torch
import torchvision
import torch.nn as nn
import os
from torchvision import models,transforms,datasets
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(model, dataloader, size, criterion, optimizer):
    model.eval()
    predictions = np.zeros(size)
    all_classes = np.zeros(size)
    all_proba = np.zeros((size,2))
    i = 0
    running_loss = 0.0
    running_corrects = 0
    for inputs,classes in dataloader:
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        loss = criterion(outputs,classes)
        _,preds = torch.max(outputs.data,1)
        # statistics
        running_loss += loss.data.item()
        running_corrects += torch.sum(preds == classes.data)
        # 将 predictions 数组从索引 i 到 i+len(classes) 这一段赋值为 preds 的值。
        predictions[i:i+len(classes)] = preds.to('cpu').numpy()
        all_classes[i:i+len(classes)] = classes.to('cpu').numpy()
        all_proba[i:i+len(classes),:] = outputs.data.to('cpu').numpy()
        i += len(classes)
        print('Testing: No. ', i, ' process ... total: ', size)
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.data.item() / size
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return predictions, all_proba, all_classes