import os
from torch.nn import Module, Linear
from torch.nn.functional import normalize
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import LFWPeople, LFWPairs
import torchvision.models as models

def get_train_dataset(p, root):
    return LFWPeople(root, split=p.train_split, transform=get_transform(p), download=True)

def get_val_dataset(p, root):
    return LFWPairs(root, split=p.test_split, transform=get_transform(p), download=True)

def get_transform(p):
    return transforms.Compose([
            transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
            transforms.Resize(p['transformation_kwargs']['resize']),
            transforms.ToTensor(),
            transforms.Normalize(**p['transformation_kwargs']['normalize']),
    ])

def get_dataloader(p, dataset):
    return DataLoader(dataset, p.batch_size, shuffle=True, num_workers=p.num_workers, pin_memory=True)

def get_val_loader(p, dataset):
    return DataLoader(dataset, batch_size=100, num_workers=1)

class ResNet(Module):
    def __init__(self, backbone, fc_layer_size):
        super(ResNet, self).__init__()

        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=True)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=True)

        in_feats = self.backbone.fc.in_features
        del self.backbone.fc
        # self.backbone.fc = Identity()
        self.backbone.fc = Linear(in_feats, fc_layer_size)
    
    def forward(self, x):
        out = self.backbone(x)
        out = normalize(out, p=2, dim=1)
        return out

def get_model(p):
    return ResNet(p.backbone, p.fc_layer_size)


def get_optimizer(p, model):
    if p.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=p.optimizer_kwargs.lr)
    elif p.optimizer == "rms":
        optimizer = optim.RMSprop(model.parameters(), lr=p.optimizer_kwargs.lr)
    elif p.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=p.optimizer_kwargs.lr)
    elif p.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=p.optimizer_kwargs.lr)
    
    return optimizer

def get_scheduler(p, optimizer):
    if p.scheduler == "reduceonplateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=p.scheduler_kwargs.patience, min_lr=p.scheduler_kwargs.min_lr, verbose=True)
    elif p.scheduler == "onecyclelr":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=p.scheduler_kwargs.max_lr, epochs=p.epochs, steps_per_epoch=9525//p.batch_size, verbose=True)
    
    return scheduler
