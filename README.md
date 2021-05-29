# Face-Unlock

This is Face-Unlock repository of IvLabs and contains the implementation of Triplet Network and FaceNet with ResNet as the backbone architechture implemented from scratch to perform one-shot and zero-shot learning on different datasets.

Our work is categorized in two parts.

- [x] [Triplet Network](triplet_network)
  - [X] Triplet Loss on MNIST
  - [x] CNN on AT&T Dataset
  - [x] ResNet on AT&T Dataset

- [x] [FaceNet](facenet)
  - [x] AT&T Dataset
  - [x] LFW Dataset

Datasets

* [The AT&T face dataset, “(formerly ‘The ORL Database of Faces’)](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/) 
  
    There are 10 different images of each of 40 distinct subjects.

    Dataset Statistics
    1. Color: Grey-scale
    2. Sample Size: 92x112
    3. #Samples: 400
   
   

* [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)
  
  Labeled Faces in the Wild is a public benchmark for face verification, also known as pair matching.

  Dataset Statistics
    1. 13233 images
    2. 5749 people
    3. 1680 people with two or more images
  

References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Understanding and visualizing ResNets](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8)
- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- [facenet-pytorch-glint360k](https://github.com/tamerthamoqa/facenet-pytorch-glint360k)

