# <a name="lorenz"></a>Lorenz Equations and Forecasting

Table of Contents
1. Introduction and Overview
2. Theoretical Background
3. Algorithm Implementation and Development 
    1. Forcasting with a Neural Network
    2. Comparing with Other Neural Networks
4. Computational Results
5. Summary and Conclusions

## Abstract

[NN.ipynb](https://github.com/marybun/machine_learning/blob/main/HW5/lorenz.ipynb) showcases forcasting with neural networks, following Homework 5. Using a feed forward neural network, we predict future outcomes of the Lorenz attractor from random initial data points, and compare the efficiency between other neural network models.

## 1. Introduction and Overview

In this notebook, we build a feed forward neural network trained using Lorenz equations with various $\rho$ values. We split the data into NN inputs and outputs (the points and the points one time step after). We test the model using two other $\rho$ values, and plot the Lorenz attractors. We then compare this model to an LSTM, RNN, and Echo State Network. The neural networks are build using PyTorch.

## 2. Theoretical Background

**Lorenz Equations** are a set of differential equations that describe chaotic behavior.

**Feed Forward Neural Network** is a type of neural network where information flows in only one direction, from input to output, without the use of feedback or loops.

**Long Short-Term Memory (LSTM)** is a type of neural network that incorporates feedback. Unlike feed forward neural networks, LSTMs store previous information to predict future outcomes.

**Recurrent Neural Network(RNN)**

**Echo State Network**

## 3. Alogrithm Implementation and Development

Import the relevant packages.

```python

```

### i. Forcasting with a Neural Network

### ii. Comparing with Other Neural Networks

## 4. Computational Results

### i. Forcasting with a Neural Network

### ii. Comparing with Other Neural Networks

## 5. Summary and Conclusions

### i. Forcasting with a Neural Network

### ii. Comparing with Other Neural Networks