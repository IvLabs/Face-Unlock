import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from itertools import combinations 
from collections import OrderedDict
from tqdm import tqdm
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

from utils import triplet_loss, get_random_triplets, count_parameters
from data import CustomDataset
from model import ResNet

PATH = '../Datasets/att_face_dataset/'
LEARNING_RATE = 0.0001
EPOCHS = 1
BATCH_SIZE = 100
CUDA = True if torch.cuda.is_available() else False
SAVE_PATH = ''


def seed_init():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def Train(model, batch, loss_fn, optimizer, cost):
    model.train()

    if CUDA:
        apn = model(batch.cuda())  
    else:
        apn = model(batch)

    optimizer.zero_grad()
    loss = loss_fn(*apn)
    cost.append(loss.item())
    loss.backward()
    optimizer.step()

    return cost[-1]


def Evaluate(model, batch):
    model.eval()
    def dist(enc1,enc2):
        return (enc1 - enc2).pow(2).sum(-1) #.pow(.5)
    with torch.no_grad():
        if CUDA:
            sample = torch.cat([model.semi_forward(imgs[0].unsqueeze(0).cuda()).cpu() for imgs in batch])
            total_enc = [model.semi_forward(img.cuda()).cpu() for img in batch]
        else:
            sample = torch.cat([model.semi_forward(imgs[0].unsqueeze(0)) for imgs in batch])
            total_enc = [model.semi_forward(img) for img in batch]
        
        pred = [torch.stack([dist(enc,sample).argmin() for enc in total_enc[i]]) for i in range(len(total_enc))]
        # acc = sum([(pred[i] == i).sum() for i in range(len(total_enc))])
        del total_enc
    # return (acc.item() / (len(batch) * 10) )
    return torch.stack(pred)


def main():
    seed_init()

    process = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.4422, std=0.1931),
    ])

    att_dataset = CustomDataset(PATH, transform=process)
    dataset = list(att_dataset)
    train = dataset[:30]
    test = dataset[30:]

    resnet18 = ResNet()
    print(f"The No. of Parameters in Model are : {count_parameters(resnet18)}")

    torch.set_grad_enabled(True)
    resnet18.train(True)

    learning_rate = LEARNING_RATE
    optimizer = optim.Adam(resnet18.parameters(), lr = learning_rate)
    torch_triplet_loss = nn.TripletMarginLoss()
    if CUDA:
        resnet18 = resnet18.cuda()

    cost = [float('inf')]
    train_acc = [0]
    test_acc = [0]
    epochs = EPOCHS

    #### TRAINING ####
    for epoch in range(epochs):

        triplets = get_random_triplets(train)
        loader = DataLoader(triplets, batch_size=BATCH_SIZE)
        steps = len(loader)
        print("Lenght of Loader:", steps)
        for i,batch in enumerate(loader):

            loss = Train(resnet18, batch, triplet_loss, optimizer, cost)
            
            pred = Evaluate(resnet18, train)
            acc1 = ( (pred == torch.arange(len(pred)).reshape(-1,1)).sum() / (len(pred)*10) ).item()
            train_acc.append(acc1)
            
            pred = Evaluate(resnet18, test)
            acc2 = ( (pred == torch.arange(len(pred)).reshape(-1,1)).sum() / (len(pred)*10) ).item()
            test_acc.append(acc2)

            if (i+1)%1==0 :
                print(f'Epoch:[{epoch+1}/{epochs}], Step:[{i+1}/{steps}]', 'Cost : {:.2f}, Train Acc: {:.2f}, Test Acc: {:.2f}'.format(loss, acc1, acc2))
                # print(f'Epoch:[{epoch+1}/{epochs}], Step:[{i+1}/87]', 'Cost : {:.2f}'.format(loss))

    plt.figure(figsize=(12,10))
    plt.title("Learning Curves")
    plt.xlabel('Total Iterations')
    plt.ylabel('Cost')
    plt.plot(np.arange(len(cost)), cost, label='cost')
    plt.plot(np.arange(len(train_acc)), train_acc, label='train_acc')
    plt.plot(np.arange(len(test_acc)), test_acc, label='test_acc')
    plt.grid(alpha=0.5)
    plt.legend()
    # plt.savefig('/content/drive/MyDrive/Colab Notebooks/siamese-orl-loss on 30classes(resnet)')
    plt.show()
    
    #### END TRAINING ####

    torch.save(resnet18.state_dict(), SAVE_PATH)

    torch.set_grad_enabled(False)
    resnet18.train(False)

    test_pred = Evaluate(resnet18, test)
    test_acc = ( (test_pred == torch.arange(len(test_pred)).reshape(-1,1)).sum() / (len(test_pred)*10) ).item()

    train_pred = Evaluate(resnet18, train)
    train_acc = ( (train_pred == torch.arange(len(train_pred)).reshape(-1,1)).sum() / (len(train_pred)*10) ).item()

    total_pred = Evaluate(resnet18, dataset)
    total_acc = ( (total_pred == torch.arange(len(total_pred)).reshape(-1,1)).sum() / (len(total_pred)*10) ).item()

    print('Train Acc: {:.2f}\nTest Acc: {:.2f}\nTotal Acc: {:.2f}'.format(train_acc, test_acc, total_acc))

if __name__ == '__main__':
    main()