import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from model import PREPROCESS as preprocess


MNIST_train = torchvision.datasets.MNIST(
    root = '/content/drive/MyDrive/Colab Notebooks/',
    download = True,
    train = True,
    transform = transforms.ToTensor()
)

# Reorganizing Data
train_dict = {i : MNIST_train.data[MNIST_train.targets == i].unsqueeze(1) / 255 for i in range(10)}
print(*[f'{imgs.shape[0]} images of Label {label} of shape {imgs.shape[1:]}' for label, imgs in train_dict.items()], sep='\n')
# for i in range(10):
#     train_dict[i] = MNIST_train.data[MNIST.targets == i]

# One Sample Image from each class
IMAGES = {0:train_dict[0][95], 1:train_dict[1][1], 2:train_dict[2][2], 3:train_dict[3][0]}
for i in [4,6,7,8]:
    IMAGES[i] = train_dict[i][1]
IMAGES[5] = train_dict[5][63]  
IMAGES[9] = train_dict[9][5]

# Visualizing Sample images
GRID = torchvision.utils.make_grid(preprocess(torch.stack([*IMAGES.values()])), nrow=5)
plt.figure(figsize=(15,5))
plt.imshow(np.transpose(GRID, (1,2,0)))

# Training only on Images of '0','1','2'
zeros_images = train_dict[0][:100]
ones_images  = train_dict[1][:100]
twos_images  = train_dict[2][:100]

train_images = torch.stack([zeros_images, ones_images, twos_images], dim=1)
# train_images.shape  # train_images[:,0,...].shape = torch.Size([100, 1, 28, 28])