# Artificial Neural Network (ANN) on MNIST

This folder contains a Jupyter Notebook demonstrating how to build, train, and evaluate a simple Artificial Neural Network using TensorFlow and Keras.

## Contents

- **`ArtificialNeuralNetwork.ipynb`**: The main notebook. It walks through:
  - Custom data loading using a `MnistDataloader` class to read the MNIST dataset from byte files.
  - Data preprocessing (normalizing pixel values to a 0.0 - 1.0 scale).
  - Building a Sequential neural network model with Keras (`tf.keras.layers.Dense`).
  - Compiling the model using the Adam optimizer and sparse categorical cross-entropy loss.
  - Training the model for 5 epochs.
  - Evaluating the model's accuracy on the test data.

## Prerequisites

To run this notebook, you will need the following Python libraries installed:
- `tensorflow`
- `matplotlib`
- `numpy`

You can install them via pip:
```bash
pip install tensorflow matplotlib numpy
```

## Dataset Structure

The notebook expects the MNIST dataset in standard idx byte format, typically extracted into the following structure:
- `train-images-idx3-ubyte/train-images-idx3-ubyte`
- `train-labels-idx1-ubyte/train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte/t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte`

Make sure the data files are present in the expected directories before running the training cells.
