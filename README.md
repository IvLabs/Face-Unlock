<!-- # Face-Unlock -->
<h1 align="center">
  Face-Unlock
</h1>

<p align="center">
<a href="https://1drv.ms/b/s!Aljy0v-8RGd7gymnNR5_Q8KYPF4S"><img src="https://img.shields.io/static/v1?label=PDF&message=Report&color=%23008080&style=for-the-badge&logo=latex&logoColor=008080" width=19% alt="PDF Report"></a>
<a href="https://bit.ly/unlockface"><img src="https://img.shields.io/badge/View%20report%20on-W%26B-%23f9c20a?style=for-the-badge" width=25% alt="View Report on W&B"></a>
<a href="https://1drv.ms/p/s!Aljy0v-8RGd7gUDq727S4vAY-gZI"><img src="https://img.shields.io/badge/View-Slides-D04423?style=for-the-badge&logo=microsoftpowerpoint&logoColor=D04423" width=20% alt="View Slides"></a>
<!-- <a href="https://github.com/IvLabs/Face-Unlock"><img src="https://img.shields.io/static/v1?label=View&message=Code&color=181717&logo=github&logoColor=181717" width=20% alt="View Slides"></a> -->
</p>

This is Face-Unlock repository of IvLabs and contains the implementation of Triplet Network and FaceNet with ResNet as the backbone architecture implemented from scratch to perform one-shot and zero-shot learning on different datasets.

<p align="center">
    <img src="https://user-images.githubusercontent.com/63636498/145682302-ef9cc6be-5289-4968-8aa0-18872684307b.gif" alt="ivpreds" >
</p>

Our work is categorized as following:

- [x] [Triplet Network](triplet_network)
  - [X] Triplet Loss on MNIST
  - [x] CNN on AT&T Dataset
  - [x] ResNet on AT&T Dataset

- [x] [FaceNet](facenet)
  - [x] AT&T Dataset
  - [x] LFW Dataset
  - [ ] Glint360k Dataset

- [ ] Real-Time Face Recognition
  - [ ] Hosting Web based implementation
  - [ ] Integrating with Rasberry Pi


## Datasets

* [The AT&T face dataset](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/) 
  
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

## ResNet

A part of this project was also to understand and implement Residual Networks from scratch which can be found in [model.py](model.py)

References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Understanding and visualizing ResNets](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8)
- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- [Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)
- [facenet-pytorch-glint360k](https://github.com/tamerthamoqa/facenet-pytorch-glint360k)

