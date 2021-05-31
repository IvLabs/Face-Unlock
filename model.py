from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, n_in, n_out, stride = 1, first_block=0):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=n_out)

        self.downsample = None
        if first_block:
            self.conv1 = nn.Conv2d(in_channels=first_block, out_channels=n_in, kernel_size=3, stride=stride, padding=1, bias=False)
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels=first_block, out_channels=n_out, kernel_size=1, stride=stride, padding=0, bias=False)),
                ('bn', nn.BatchNorm2d(num_features=n_out))
            ]))

    def forward(self, x):
        x_shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            x_shortcut = self.downsample(x_shortcut)

        x = x + x_shortcut
        x = self.relu(x)

        return x



class ResidualBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(
        self, n_in, n_out, stride = 1, first_block=0):
        super(ResidualBottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=n_out, out_channels=n_in, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(n_in)
        self.conv2 = nn.Conv2d(in_channels=n_in, out_channels=n_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_in)
        self.conv3 = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if first_block:
            self.conv1 = nn.Conv2d(in_channels=first_block, out_channels=n_in, kernel_size=1, stride=stride, padding=0, bias=False)
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels=first_block, out_channels=n_out, kernel_size=1, stride=stride, padding=0, bias=False)),
                ('bn', nn.BatchNorm2d(num_features=n_out))
            ]))

    def forward(self, x):
        x_shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample:
            x_shortcut = self.downsample(x_shortcut)

        x = x + x_shortcut
        x = self.relu(x)

        return x



class ResNet(nn.Module):
    """
    ResNet18  = ResNet(layers=[2,2,2,2])
    ResNet34  = ResNet(layers=[3,4,6,3])
    ResNet50  = ResNet(layers=[3,4,6,3], bottleneck=True)
    ResNet101 = ResNet(layers=[3,4,23,3],bottleneck=True)
    ResNet152 = ResNet(layers=[3,8,36,3],bottleneck=True)
    """
    def __init__(self, layers = [2, 2, 2, 2], num_classes = 1000, inplanes = 3, bottleneck=False):
        super(ResNet, self).__init__()

        self.inplanes = 64
        block = ResidualBlock
        self.n = sum(layers)*2 + 2
        if bottleneck:
            block = ResidualBottleneckBlock
            self.n = sum(layers)*3 + 2

        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)

        self.layer2 = self._make_layer(block, 128, layers[1])

        self.layer3 = self._make_layer(block, 256, layers[2])

        self.layer4 = self._make_layer(block, 512, layers[3])

        self.avgpool = nn.AvgPool2d(kernel_size=7)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(in_features=512*block.expansion,
                            out_features=num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, num_residuals, stride = 2) -> nn.Sequential:
        block_layers = []
        first_block = self.inplanes if self.inplanes != channels*block.expansion else 0
        block_layers.append(
            (f'block{1}', block(channels, channels*block.expansion, stride, first_block)))

        for i in range(1, num_residuals):
            block_layers.append(
                (f'block{i+1}', block(channels, channels*block.expansion)))

        self.inplanes = channels*block.expansion
        return nn.Sequential(OrderedDict(block_layers))

    def _get_name(self):
        return self.__class__.__name__ + str(self.n)

    def semi_forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(input=x, start_dim=1)
        x = self.fc(x)
        return x

    def forward(self, triplet):
        out = self.semi_forward(triplet)
        out = F.normalize(out, p=2, dim=1)
        return out



def ResNet18(**kwargs):
    return ResNet(layers=[2,2,2,2], **kwargs)

def ResNet34(**kwargs):
    return ResNet(layers=[3,4,6,3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(layers=[3,4,6,3], bottleneck=True, **kwargs)

def ResNet101(**kwargs):
    return ResNet(layers=[3,4,23,3],bottleneck=True, **kwargs)

def ResNet152(**kwargs):
    return ResNet(layers=[3,8,36,3],bottleneck=True, **kwargs)



# ref : https://arxiv.org/abs/1512.03385
# ref : https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
# ref : https://d2l.ai/chapter_convolutional-modern/resnet.html
# ref : https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L144
