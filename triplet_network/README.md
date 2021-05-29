# Triplet-Network using Pytorch

Face Recognition is genarlly a one-shot learning task. One shot learning is a classification task where the model should learn from one example of given class and be able to recognize it in the future. 

The loss function used is triplet-loss function which attempts to minimize the distance between embeddings belonging to same class while maximize the distance between embeddings of different classes.

<!-- $L = \sum_{i}^{N} [\; ||f(x^a_i) - f(x^p_i)||_2^2 \;-\; ||f(x^a_i)- f(x^n_i)||_2^2 + \alpha\;]_+$ -->
<img src="https://latex.codecogs.com/svg.latex?\bg_white&space;L&space;=&space;\sum_{i}^{N}&space;[&space;||f(x^a_i)&space;-&space;f(x^p_i)||_2^2&space;\;-\;&space;||f(x^a_i)-&space;f(x^n_i)||_2^2&space;&plus;&space;\alpha\;]_&plus;" title="L = \sum_{i}^{N} [ ||f(x^a_i) - f(x^p_i)||_2^2 \;-\; ||f(x^a_i)- f(x^n_i)||_2^2 + \alpha\;]_+" />

<img src="Triplet%20Net%20ATT/media/triplet_net.png" width="70%">

## [Triplet Loss on MNIST](Triplet%20Loss%20MNIST)

Triplet loss was implemented on MNIST as a part of learning the new approach.

### Training

The model was trained only on `100 images of classes 0, 1 and 2`, rest images were unknown to the model.

|    Param   |     Value    |
|:----------:|:------------:|
|  Optimizer |     Adam     |
|    Loss    | Triplet Loss |
| Batch_size |      10      |
|   Epochs   |       5      |

### Results

|  Classes |    0   |    1   |    2   |    3   |    4   |    5   |    6   |    7   |    8   |    9   |
|:--------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|  Correct |  5804  |  6648  |  5830  |  5877  |  5830  |  5274  |  5908  |  5589  |  5777  |  5849  |
|   Total  |  5923  |  6742  |  5958  |  6131  |  5842  |  5421  |  5918  |  6265  |  5851  |  5949  |
| Accuracy | 97.99% | 98.60% | 97.85% | 95.85% | 99.79% | 97.28% | 99.83% | 89.20% | 98.73% | 98.31% |


## [Triplet Net on AT&T Dataset](Triplet%20Net%20ORL)

The dataset was first used to train model with simple CNN, later ResNets implemented from scratch were used for different aproaches.
Offline triplet selection method was implied during the training and turned out that it's not very optimum choice.

## Summary
|          Architechture          | No. of learnable  parameters |   Training Set   |     Test Set     | Epochs |  Learning  Rate | Optimizer | Train Accuracy | Test Accuracy |
|:-------------------------------:|:----------------------------:|:----------------:|:----------------:|:------:|:---------------:|:---------:|:--------------:|:-------------:|
|            Plain CNN            |           4,170,400          |   75% (300/400)  |   25% (100/400)  |   16   | 10<sup>-4</sup> |    Adam   |     92.67%     |     88.00%    |
| ResNet-18 (Face-Identification) |          11,235,904          | 70% (38x7/38x10) | 30% (38x3/38x10) |    8   | 20<sup>-4</sup> |    Adam   |     99.62%     |     94.73%    |
|       ResNet-18 (One-Shot)      |          11,235,904          |   75% (300/400)  |   25% (100/400)  |   20   | 10<sup>-4</sup> |    Adam   |     82.00%     |     87.00%    |
|            ResNet-26            |          17,728,064          |   75% (300/400)  |   25% (100/400)  |   20   | 20<sup>-4</sup> |    Adam   |     93.00%     |     69.00%    |

## Plots
<p align="center">
  <img src = "Triplet%20Net%20ATT/media/siamese-orl-loss%20on%2030classes.png" width="40%"/>
  <img src = "Triplet%20Net%20ATT/media/siamese-orl-loss%20on%2030classes(resnet).png" width="40%"/>
  <img src = "Triplet%20Net%20ATT/media/siamese-orl-loss%20on%2038classes(resnet18)6.png" width="40%"/>
  <img src = "Triplet%20Net%20ATT/media/siamese-orl-accuracy%20on%2030classes(resnet)1.png" width="40%"/>
</p>
