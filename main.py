import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from itertools import combinations 
from collections import OrderedDict
from torch.utils.data import dataset
from tqdm import tqdm #, tqdm_notebook
from tqdm.notebook import tqdm as tqdm_notebook
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from datasets import AttDataset

PATH = "C:\\Users\\abdu\Projects\\Siamese-Triplet\\Datasets\\att_face_dataset"
dataset = AttDataset(PATH)

