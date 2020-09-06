# Developing an Image Classifier with Deep Learning
This project demonstrates how to make use of pre-trained deep learning models, and then enhancing them with custom feed-forward networks. This is implemented in Python3 using PyTorch, and supports GPU-based training. The project is trained and tested on an image dataset of flowers composed of 102 variants, but the project supports the ability to train a classifier on any image dataset.

## Project structure

### 1. Interactive Jupyter Notebook
This highlights the data processing and model training components, with detailed interactive cells to explain the process. The SqueezeNet 1.1 model is used, and the number of output features are configured to match the number of classes in the input dataset.

### 2. Console Application
This builds off the components demonstrated in the notebook and provides a highly customisable application allowing for modification of:

* Input dataset (directory structure must match)
* Input class mappings
* Model architecture (currently supports AlexNet, ResNet18 and SqueezeNet 1.1 but can be expanded)
* Number of hidden units in model feed-forward network
* Learning rate
* Training epochs
* Loading pre-trained model checkpoint
* Number of predicted class results
* GPU-based training / prediction

The ResNet18 model implements a custom feed-forward network using linear, relu, and dropout layers that can be modified as required.

## Package Requirements
Versions are currently agnostic. Install a combination is simply compatible:

* Python3
* PyTorch (with CUDA if GPU is available)
* NumPy
* Pandas
* Matplotlib
* Seaborn

## Console Application Arguments:
The console application can be used to either train a model, or analyse a specified input.

### Train:
Basic usage: 

```python train.py [input_data_directory]```

Optional arguments:

* ```--save_dir [checkpoint_save_directory]```
* ```--arch [model_architecture]```
* ```--learning_rate [optimizer_learn_rate]```
* ```--hidden_units [features_in_feed_forward_layer]```
* ```--epochs [training_epochs]```
* ```--gpu```

Example argument usage:

```python train.py flowers --arch resnet18 --learning_rate 0.001 --epochs 10 --gpu```

### Predict:
Basic usage: 

```python predict.py [path_to_image] [saved_checkpoint]```

Optional arguments:

* ```--top_k [number_of_returned_predictions]```
* ```--category_names [json_file_of_class_mappings]```
* ```--gpu```

Example argument usage:

```python predict.py flowers/test/1/image_06743.jpg checkpoint_squeezenet1_1.pth --gpu --top_k 10 --category_names cat_to_name.json```
