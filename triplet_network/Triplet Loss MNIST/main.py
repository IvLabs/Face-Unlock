import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
 
from data import train_dict as mnist_images
from data import IMAGES, train_images
from utils import triplet_loss, get_random_triplets, plot_embeddings
from model import Model


def weights_init(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')  # He-initialization
        # ref : https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        
    # Reference : https://discuss.pytorch.org/t/weight-initilzation/157

def main():
    model = Model()
    model.apply(weights_init)

    learning_rate = 0.001
    lossfn = triplet_loss
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    loader = torch.utils.data.DataLoader(train_images, batch_size=10)
    print('Lenght of Loader',len(loader))

    ###  TRAINING ###
    cost = []
    epochs = 5
    for epoch in range(epochs):
        for batch in loader:
            zeros_encodings = model(batch[:,0,...])
            ones_encodings  = model(batch[:,1,...])
            twos_encodings  = model(batch[:,2,...])

            triplets = get_random_triplets([zeros_encodings, ones_encodings, twos_encodings], [0, 1, 2])

            anchor   = triplets[:,0]
            positive = triplets[:,1]
            negative = triplets[:,2]

            optimizer.zero_grad()
            loss = lossfn(anchor, positive, negative, margin=1)
            cost.append(loss)
            
            loss.backward()
            optimizer.step()

            print(f'Epoch:[{epoch+1}/{epochs}] , Cost : {loss.item()}')

    plt.figure()
    plt.xlabel('Total Iterations')
    plt.ylabel('Cost')
    plt.plot(np.arange(epochs*len(loader)), cost)
    plt.show()

    #### END TRAINING ####

    ### PLOT EMBEDDINGS ####
    # enc0 = []
    # enc1 = []
    # enc2 = []
    # for batch in tqdm(loader):
    #     enc0.append(model(batch[:,0,...]))
    #     enc1.append(model(batch[:,1,...]))
    #     enc2.append(model(batch[:,2,...]))
    # enc0 = torch.cat(enc0)
    # enc1 = torch.cat(enc1)
    # enc2 = torch.cat(enc2)
    # plot_embeddings([enc0, enc1, enc2], [0,1,2])
    ########

    #### TESTING ####
    IMAGES_ENC = {k:model(v.unsqueeze(0)) for k,v in IMAGES.items()}

    correct = {i:0 for i in range(10)}
    for k , value in mnist_images.items():
        vload = torch.utils.data.DataLoader(value, batch_size=50)
        for v in tqdm(vload, desc=f"Class : {k}", position=0, leave=True):
            correct[k] += ((model(v) - IMAGES_ENC[k]).pow(2).sum(-1) < 12.6).sum()

    for i in range(10):
        c = correct[i].item()
        t = mnist_images[i].shape[0]
        print(f'Class {i} : Correct = {c} out of {t} . Accuracy = {c / t * 100} %')

    '''    
    Class 0 : Correct = 5804 out of 5923 . Accuracy = 97.9908829984805 %
    Class 1 : Correct = 6648 out of 6742 . Accuracy = 98.60575496885198 %
    Class 2 : Correct = 5830 out of 5958 . Accuracy = 97.85162806310842 %
    Class 3 : Correct = 5877 out of 6131 . Accuracy = 95.85711955635297 %
    Class 4 : Correct = 5830 out of 5842 . Accuracy = 99.79459089352962 %
    Class 5 : Correct = 5274 out of 5421 . Accuracy = 97.28832318760377 %
    Class 6 : Correct = 5908 out of 5918 . Accuracy = 99.83102399459277 %
    Class 7 : Correct = 5589 out of 6265 . Accuracy = 89.2098962490024 %
    Class 8 : Correct = 5777 out of 5851 . Accuracy = 98.73525893009742 %
    Class 9 : Correct = 5849 out of 5949 . Accuracy = 98.31904521768364 %
    '''

    #### END TESTING ####

    # Saving the state_dict of Model
    torch.save(model.state_dict(), '/content/Triplet Loss/tripletLoss_on_MNIST')

if __name__ == '__main__':
    main()