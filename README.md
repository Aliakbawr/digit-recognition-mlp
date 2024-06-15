# digit-recognition-mlp

# Neural Network Project: Analyzing MNIST Data

This project explores various aspects of training neural networks using the MNIST dataset. It includes experiments on the effects of different model architectures, optimization algorithms, learning rates, and activation functions. The code is implemented in a Jupyter Notebook using TensorFlow and Keras.

## Table of Contents
1. [Importing Libraries](#importing-libraries)
2. [Loading and Preprocessing Data](#loading-and-preprocessing-data)
3. [Effect of Number of Layers](#effect-of-number-of-layers)
4. [Effect of Number of Neurons](#effect-of-number-of-neurons)
5. [Effect of Optimization Algorithms](#effect-of-optimization-algorithms)
6. [Effect of Learning Rate](#effect-of-learning-rate)
7. [Overfitting and Underfitting](#overfitting-and-underfitting)
8. [Stopping Criteria](#stopping-criteria)
9. [Effect of Activation Functions](#effect-of-activation-functions)
10. [Conclusion](#conclusion)

## Importing Libraries

The following libraries are essential for loading the MNIST dataset, building and training neural networks, and visualizing results.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad
from tensorflow.keras.callbacks import EarlyStopping
```

## Loading and Preprocessing Data

Load the MNIST dataset and preprocess it by normalizing the pixel values and converting the labels to categorical format.

```python
# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

## Effect of Number of Layers

Analyze how the number of layers in a neural network affects the loss and accuracy.

```python
def build_model(num_layers):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for _ in range(num_layers):
        model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_effect_of_layers(x_train, y_train, x_test, y_test, max_layers=5, epochs=10):
    # Implementation of the function to plot the effect of layers
```

## Effect of Number of Neurons

Investigate the impact of varying the number of neurons in the hidden layers.

```python
def build_model(num_layers, num_neurons):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for _ in range(num_layers):
        model.add(Dense(num_neurons, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_effect_of_neurons(x_train, y_train, x_test, y_test, neurons_list, num_layers=2, epochs=10):
    # Implementation of the function to plot the effect of neurons
```

## Effect of Optimization Algorithms

Evaluate different optimization algorithms and their effect on training and validation performance.

```python
def build_model(num_layers, optimizer):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for _ in range(num_layers):
        model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_effect_of_optimizers(x_train, y_train, x_test, y_test, num_layers=3, epochs=10):
    # Implementation of the function to plot the effect of optimizers
```

## Effect of Learning Rate

Study the effect of different learning rates on model performance.

```python
def train_model_with_lr(x_train, y_train, x_test, y_test, learning_rate, epochs=20):
    # Implementation of the function to train models with different learning rates
```

## Overfitting and Underfitting

Compare models that underfit and overfit the data to understand these concepts.

```python
def train_underfitting_model(x_train, y_train, x_test, y_test, epochs=5):
    # Implementation of the function to train an underfitting model

def train_overfitting_model(x_train, y_train, x_test, y_test, epochs=20):
    # Implementation of the function to train an overfitting model
```

## Stopping Criteria

Implement and compare models using different stopping criteria such as fixed epochs and early stopping based on validation loss or accuracy.

```python
def train_fixed_epochs_model(x_train, y_train, x_test, y_test, epochs=5):
    # Implementation of the function to train a model with fixed epochs

def train_early_stopping_loss_model(x_train, y_train, x_test, y_test, patience=3):
    # Implementation of the function to train a model with early stopping on validation loss

def train_early_stopping_accuracy_model(x_train, y_train, x_test, y_test, patience=3):
    # Implementation of the function to train a model with early stopping on validation accuracy
```

## Effect of Activation Functions

Analyze the effect of different activation functions (ReLU, Sigmoid, Tanh) on the training and validation performance.

```python
def train_relu_model(x_train, y_train, x_test, y_test, epochs=5):
    # Implementation of the function to train a model with ReLU activation

def train_sigmoid_model(x_train, y_train, x_test, y_test, epochs=5):
    # Implementation of the function to train a model with Sigmoid activation

def train_tanh_model(x_train, y_train, x_test, y_test, epochs=5):
    # Implementation of the function to train a model with Tanh activation
```

## Conclusion

The project demonstrates various experiments to understand the impact of different neural network hyperparameters and configurations. The results provide insights into how changes in model architecture, optimization algorithms, learning rates, and activation functions affect the performance of neural networks on the MNIST dataset.
... 
 For reviewing code, open dig_rec.ipynb.
 For reviewing Code analysis, open documentation.pdf.
 For reviewing the final classifier, open dig-class.ipynb.