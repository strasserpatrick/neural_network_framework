# Core Library for Neural networks

This library serves as a core library for the gesture control task. Despite that, it can be used in standalone, as the examples in `examples/` folder show.

## Installation

```bash
pip install -r requirements.txt
```

## Usage example (Red wine quality prediction)

```bash
python examples/winequality-binary-classification.py
```

## Component Explanation:

### Dataset: 
Abstraction for wrapping arbitrary data formats into our dataset class for common handling. `NumpyDataset` is a concrete implementation for numpy arrays.

### Dataloader: 
Wrapper for dataset, which handles indexing, shuffling, and other data loading related tasks.

### Optimizer:
Collection of different optimizers for the backpropagation algorithm. Currently we only implemented SGD and SGD with Momentum.

### Functions:
A collection of activation and loss functions.

### Metrics: 
A collection of evaluation metrics for the model.

### Transforms:
A collection of data transforms, such as OneHotEncoding, Minority Sampling, MinMaxScaler, Standardizer, etc.

### PCA:
A class for performing PCA on the dataset.

### Layers:
An abstraction for layers in the neural network used for models. Concrete implementations for fully-connected layers and dropout layers are provided.

### Model:
Sequential model holds a sequence of layers. The output of the previous layer is the input of the next layer. The model can be trained and evaluated on the dataset.

### Trainer: 
Helper class that performs training and evaluation on a model, a dataset and an optimizer. Additionally, it is able to perform sanity checks on the model. Models hyperparameters like learning rate and epochs are set here.

## Features

- [x] Fully connected layers
- [x] Dropot layers
- [x] Multiple Loss and Activation functions
- [x] Optimizers (SGD, Momentum)
- [x] PCA
- [x] Evaluation metrics (Accuracy, Precision, Recall, F1, confusion matrix)
- [x] Callback functions for validation score
- [x] ...
