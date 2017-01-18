# Implementation of a Multimodal Deep Boltzmann Machine with Tensorflow.
This is a project realized in the course of Probabilistic Graphical Model, Object Recognition and Computer Vision. 

The goal is to implement a Multi-DBM model with Tensorflow. This is inspired from the [paper]( http://jmlr.org/papers/volume15/srivastava14b/srivastava14b.pdf) of Nitish Srivastava et al appeared at NIPS2012. 



The code is built on the top of the [Deep-Learning-Tensorflow](https://github.com/blackecho/Deep-Learning-TensorFlow) 

* Convolutional Network
* Recurrent Neural Network (LSTM)
* Restricted Boltzmann Machine
* Deep Belief Network
* Deep Autoencoder as stack of RBMs
* Denoising Autoencoder
* Stacked Denoising Autoencoder
* Deep Autoencoder as stack of Denoising Autoencoders
* MultiLayer Perceptron
* Logistic Regression

### Installation

#### Through pip:

    pip install yadlt

You can learn the basic usage of the models by looking at the ``command_line/`` directory. Or you can take a look at the [documentation](http://deep-learning-tensorflow.readthedocs.io/en/latest/).

**Note**: the documentation is still a work in progress for the pip package, but the package usage is very simple. The classes have a sklearn-like interface, so basically you just have to create the object
(e.g. `sdae = StackedDenoisingAutoencoder()`) and call the fit/predict methods, and the pretrain() method if the model supports it
(e.g. `sdae.pretrain(X_train, y_train)`, `sdae.fit(X_train, y_train)` and `predictions = sdae.predict(X_test)`)

#### Through github:

* cd in a directory where you want to store the project, e.g. ``/home/me``
* clone the repository: ``git clone https://github.com/blackecho/Deep-Learning-TensorFlow.git``
* ``cd Deep-Learning-TensorFlow``
* now you can configure the software and run the models (see the [documentation](http://deep-learning-tensorflow.readthedocs.io/en/latest/))!

### Documentation:

You can find the documentation for this project at this [link](http://deep-learning-tensorflow.readthedocs.io/en/latest/).

### Models TODO list
* Multimodal Deep Boltzmann Machine
* Variational Autoencoders
* Deep Q Reinforcement Learning
