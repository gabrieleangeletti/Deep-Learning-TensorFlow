# Deep Belief Network and Restricted Boltzmann Machine Python Implementation

The MNIST dataset and MNIST dataset loader script are taken from https://github.com/sorki/python-mnist

Requirements:
- numpy
- scikitlearn

Example Usage on MNIST:

- config.py:
Configuration file, used to set the training parameters.

- main.py:
Main file used for training.

- Set all the parameters in config.py.
- run python main.py

 main:
     usage: main.py [-h] -n Neural Network type [-o output file]
    
    Execute RBMs and DBNs on the MNIST digits dataset.
    
    optional arguments:
    
      -h, --help            show this help message and exit
      
      -n Neural Network type
                            the kind of NN to run: rbm (binary), mrbm
                            (multinomial), grbm (gaussian),crbm (classification
                            rbm), cmrbm (class. multinomial), cgrbm (class.
                            gaussian) dbn (deep net)...
                            
      -o output file        output file to store the configuration of trained NNs.
                            If a file exists,it will be overwritten

