from model import LeNet
import torch
import torchvision
import torch.nn as nn
import os
from torchvision import models,transforms,datasets

# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, size, epochs, criterion, optimizer):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        count = 0

        for inputs,classes in dataloader:
            inputs = inputs.to(device)
            classes = classes.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _,preds = torch.max(outputs.data,1)
            # statistics
            running_loss += loss.data.item()
            running_corrects += torch.sum(preds == classes.data)
            count += len(inputs)
            print('Training: No. ', count, ' process ... total: ', size)
        epoch_loss = running_loss / size
        epoch_acc = running_corrects.data.item() / size
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


