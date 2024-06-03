import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用类型于LeNet 的网络来分类猫与狗
# 输入： 3 * 256 * 256
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 3 * 256 * 256
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, bias=False)
        # 16 * 254 * 254
        self.batch1 = nn.BatchNorm2d(24)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 16 * 127 * 127

        self.conv2 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, bias=False)
        # 24 * 125 * 125
        self.batch2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 24 * 62 * 62
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False)
        # 24 * 60 * 60
        self.batch3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # 24 * 30 * 30
        self.fc1 = nn.Linear(32 * 30 * 30, 72)
        self.fc2 = nn.Linear(72, 36)
        self.fc3 = nn.Linear(36, 2)


    def forward(self, x):

        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x



class BasicNetBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride):
        super(BasicNetBlock, self).__init__()

        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, bias=False,stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion, bias=False, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        if stride != 1 or in_channel != out_channel * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * self.expansion)
            )
        else:
            self.shortcut = nn.Identity()


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.in_channel=16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(BasicNetBlock, 16, 3, 1)
        self.layer2 = self._make_layer(BasicNetBlock, 32, 3, 2)
        self.layer3 = self._make_layer(BasicNetBlock, 32, 3, 2)
        self.fc1 = nn.Linear(in_features=32 * BasicNetBlock.expansion *14 *14, out_features=2)

    def forward(self, x):
        # 3*228*228
        x = F.relu(self.bn1(self.conv1(x)))
        # 16*226*226
        x = self.layer1(x)
        # 4*16*226*226
        x = self.layer2(x)
        # 4*32*113*113
        x = self.layer3(x)
        # 4*32*56*56
        x = F.max_pool2d(x, 4)
        # 4*32*14*14
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def _make_layer(self, block, out_channel, blockNumber, stride):
        layers = []
        layers.append(block(self.in_channel, out_channel, stride))
        self.in_channel = out_channel * block.expansion
        for stride in range(blockNumber - 1):
            layers.append(block(self.in_channel, out_channel, 1))

        return nn.Sequential(*layers)



class TempNet(nn.Module):
    def __init__(self):
        super(TempNet, self).__init__()
        self.fc1 = BasicNeXtBlock(3, 16, 1, 1)
        self.fc2 = BasicNeXtBlock(64, 24, 1, 1)
        self.fc3 = nn.Linear(24, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), -1)
        return x
    
