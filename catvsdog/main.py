from train import train
from test import test
from model import LeNet,ResNet,TempNet
import torch
import torchvision
import torch.nn as nn
import os
from torchvision import models,transforms,datasets
from PIL import Image
import pandas as pd


transform = transforms.Compose([
    transforms.Resize(228),
    transforms.CenterCrop(228),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),

])

data_dir = './cat_dog'

dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=transform)
         for x in ['train', 'val']}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes

loader_train = torch.utils.data.DataLoader(dsets['train'], batch_size=16, shuffle=True, num_workers=12)
loader_valid = torch.utils.data.DataLoader(dsets['val'], batch_size=5, shuffle=False, num_workers=6)

model = ResNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model_pt_path = "./ResNet_model.pt"

def train_and_save_mode():
    print("开始训练")

    #train(model, loader_valid, dset_sizes['val'], 1, criterion, optimizer)
    train(model, loader_train, dset_sizes['train'], 20, criterion, optimizer)
    
    test(model, loader_valid, dset_sizes['val'], criterion, optimizer)

    torch.save(model.state_dict(), model_pt_path)

def load_and_compute():
    model.load_state_dict(torch.load(model_pt_path))
    model.eval()
    # 准备结果列表
    results = []

    # 遍历文件夹中的图片\
    image_folder = "./cat_dog/test"
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 根据需要修改文件扩展名
            print(filename)
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)
            input_tensor = transform(image)
            input_batch = input_tensor.unsqueeze(0)  # 创建一个batch

            with torch.no_grad():
                output = model(input_batch)
                _, predicted = torch.max(output, 1)
                result = predicted.item()
            file_id = os.path.splitext(filename)[0]
            # 将结果添加到列表
            results.append({'file_id': file_id, 'result': result})
    results.sort(key=lambda x: int(x['file_id']))
    # 将结果写入CSV文件
    df = pd.DataFrame(results)
    df.to_csv("./result.csv", index=False)


if __name__ == '__main__':
    train_and_save_mode()






