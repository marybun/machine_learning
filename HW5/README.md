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

[lorenz_NN.ipynb](https://github.com/marybun/machine_learning/blob/main/HW5/lorenz_NN.ipynb) showcases forcasting with neural networks, following Homework 5. Using a feed forward neural network, we predict future outcomes of the Lorenz attractor from random initial data points, and compare the efficiency between other neural network models.

## 1. Introduction and Overview

In this notebook, we build a feed forward neural network trained using Lorenz equations with various $\rho$ values. We split the data into NN inputs and outputs (the points and the points one time step after). We test the model using two other $\rho$ values, and plot the Lorenz attractors. We then compare this model to an LSTM, RNN, and Echo State Network. The neural networks are build using PyTorch.

## 2. Theoretical Background

**Lorenz Equations** are a set of three differential equations by Edward Lorenz that describe chaotic behavior. They are given by:

$\frac{dx}{dt} = \sigma(y-x)$

$\frac{dy}{dt} = x(\rho-z)-y$

$\frac{dz}{dt} = xy - \beta z$

**Feed Forward Neural Network** is a type of neural network where information flows in only one direction, from input to output, without the use of feedback or loops.

**Recurrent Neural Network (RNN)** is a type of neural network used for processing sequential data. It keeps a memory of previous information to make predictions.

**Long Short-Term Memory (LSTM)** is a type of RNN that uses gates to control the flow of information. These gates allow the network to remember and forget specific information.

**Echo State Network** is a type of RNN that is much simpler than other models. It has a fixed randomly initialized layer called the "reservoir" that acts as the memory of the network.

## 3. Alogrithm Implementation and Development

Import the relevant packages.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
```

### i. Forcasting with a Neural Network

First, we set up the parameters and Lorenz equations.
```python
plt.rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [12, 12]

dt = 0.01
T = 8
t = np.arange(0,T+dt,dt)
beta = 8/3
sigma = 10.0
rho =28
# Define the Lorenz equations
def lorenz_deriv(x_y_z, t0, rho=rho, sigma=sigma, beta=beta):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
```
We create a function to generate initial data points and plug them into the lorenz equations using rho values of 10, 28, and 40. We solve for the training data values, plot them, and concatenate them.

```python
rho_train = [10, 28, 40]
np.random.seed(123) ## Set once before multiple calls to random() function
# Define a function to generate training data for the neural network
def generate_data(delta_t, rho):
    # Generate random initial conditions
    x0 = -15 + 30 * np.random.random((100, 3))

    # Integrate the Lorenz equations to generate training data
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args = (rho, sigma, beta))
                    for x0_j in x0])

    fig,ax = plt.subplots(1,1,subplot_kw={'projection': '3d'})
    nn_input = np.zeros((100*(len(t)-1),3))
    nn_output = np.zeros_like(nn_input)

    for j in range(100):
        nn_input[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,:-1,:]
        nn_output[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,1:,:]
        x, y, z = x_t[j,:,:].T
        ax.plot(x, y, z,linewidth=1)
        ax.scatter(x0[j,0],x0[j,1],x0[j,2],color='r')
        plt.title('Lorenz Attractor for rho = ' + str(rho))

    ax.view_init(18, -113)
    plt.show()
    return x0, nn_input, nn_output

# Generate training data using rho_train values
nn_input_train, nn_output_train = [], []
for rho_train_value in rho_train:
    x0, input_data, output_data = generate_data(dt, rho_train_value)
    nn_input_train.append(input_data)
    nn_output_train.append(output_data)

nn_input_train = np.concatenate(nn_input_train)
nn_output_train = np.concatenate(nn_output_train)
print("nn_input_train:", nn_input_train.shape)
print("nn_output_train:", nn_output_train.shape)
```

We create a TrainDataset class to allow us to train the NNs in batches.
```python
from torch.utils.data import DataLoader, Dataset

class TrainDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input = torch.tensor(input_data, dtype=torch.float32)
        self.output = torch.tensor(output_data, dtype=torch.float32) 
        self.n_samples=self.input.shape[0]
    def __getitem__(self, index):
        return self.input[index], self.output[index]
    def __len__(self):
        return self.n_samples
```

We then make a function to train a given NN model.

```python
# Create the neural network and optimizer
def train_NN(model, input_train, output_train, num_epochs = 100, batch_size = 128):
    print(f'batch_size: {batch_size}')
    dataset = TrainDataset(input_train, output_train)   
    total_samples = len(dataset)
    print(f'total_samples: {total_samples}')
    num_iterations = math.ceil(total_samples/batch_size)
    print(f'n_iterations: {num_iterations}')

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
   
    optimizer = torch.optim.Adam(model.parameters())
    mse=np.zeros(num_epochs)
    loss_fn = nn.MSELoss()
    
    for epoch in range(num_epochs):
        for i, (inputs, outputs) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = model(inputs)
            loss = loss_fn(pred, outputs)
            loss.backward()
            optimizer.step()
            mse[epoch]=loss
            if((i+1) % 100 == 0):
                print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{num_iterations},  MSELoss={mse[epoch]:0.4f}')
    plt.plot(range(num_epochs),mse)
    plt.title("Error")
    plt.ylabel("MSE")
    plt.xlabel("Epoch")
    return model    
```

We make a 3 layer feed forward NN and train it over the rho_train values.

```python
# Define a neural network to predict the next state given the current state
class LorenzNN(nn.Module):
    def __init__(self):
        super(LorenzNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# Train the neural network using all rho_train values
model = LorenzNN()
model = train_NN(model, input_train=nn_input_train, output_train=nn_output_train,num_epochs=100)
```

We define a function to predict and show how well the model performs.

```python
def model_predict(model, rho):
    x0 = -15 + 20 * np.random.random((2,3))
    x_t = integrate.odeint(lorenz_deriv, x0[0,:], t, args = (rho, sigma, beta))

    ynn = np.zeros((len(t),3))
    ynn[0,:]=x0[0,]
    print("x0     :", ynn[0,:])
    with torch.no_grad():
        y0 = torch.tensor(ynn[0,:], dtype=torch.float32)
        for j in range(len(t)-1):
            y1 = model(y0)
            y0 = y1
            ynn[j+1,:] = y1.detach().numpy()
            if((j+1) % 100 == 0):
                print("predict:",ynn[j+1,:])
                print("actual :", x_t[j,])
    print(x0.shape)
    print(ynn.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_t[:,0], x_t[:,1], x_t[:,2], linewidth=1)
    ax.plot(ynn[:,0],ynn[:,1],ynn[:,2],'g.')
    ax.scatter(x0[0,0],x0[0,1],x0[0,2],'r')
    plt.title(type(model).__name__ + ' Prediction of Lorenz Attractor, rho=' + str(rho))
    ax.view_init(18, -113)
    plt.show()

    fig = plt.figure()

    ax = fig.add_subplot(3,1,1)
    ax.plot(t, x_t[:,0], linewidth=1)
    ax.plot(t, ynn[:,0],'g.')
    plt.title(type(model).__name__ + ' Prediction of Lorenz Attractor, rho= ' + str(rho))
    ax.set_ylabel("x(t)")

    ax = fig.add_subplot(3,1,2)
    ax.plot(t, x_t[:,1], linewidth=1)
    ax.plot(t, ynn[:,1],'g.')
    ax.set_ylabel("y(t)")
            
    ax = fig.add_subplot(3,1,3)
    ax.plot(t, x_t[:,2], linewidth=1)
    ax.plot(t, ynn[:,2],'g.')
    ax.set_ylabel("z(t)")
    ax.set_xlabel("t")
    plt.show()
```

We then test the model over 2 rho values, 17 and 35.

```python
np.random.seed(133) ## Set once before multiple calls to random() function
model_predict(model, rho = 17)
model_predict(model, rho = 35)
```

### ii. Comparing with Other Neural Networks

**LSTM**

We build an LSTM model and train it.

```python
# Define a neural network to predict the next state given the current state
class LorenzLSTM(nn.Module):
    def __init__(self):
        super(LorenzLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        x = self.fc(h[-1])
        return x
    
# Train the neural network using all rho_train values
model = LorenzLSTM()
model = train_NN(model, input_train=nn_input_train, output_train=nn_output_train, num_epochs=100)
```

LSTM does not work with 1D Tensors, so we unsqueeze them. We then plot the Lorenz attractor and other figures.

```python
def LSTM_predict(model, rho):
    x0 = -15 + 20 * np.random.random((2, 3))
    x_t = integrate.odeint(lorenz_deriv, x0[0, :], t, args=(rho, sigma, beta))

    ynn = np.zeros((len(t), 3))
    ynn[0, :] = x0[0, :]
    print("x0     :", ynn[0, :])
    with torch.no_grad():
        y0 = torch.tensor(ynn[0, :], dtype=torch.float32)
        for j in range(len(t) - 1):
            y1 = model(y0.unsqueeze(0)) 
            y0 = y1.squeeze()
            ynn[j + 1, :] = y1.detach().numpy().squeeze()
            if (j + 1) % 100 == 0:
                print("predict:", ynn[j + 1, :])
                print("actual :", x_t[j, :])
    print(x0.shape)
    print(ynn.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_t[:, 0], x_t[:, 1], x_t[:, 2], linewidth=1)
    ax.plot(ynn[:, 0], ynn[:, 1], ynn[:, 2], "g.")
    ax.scatter(x0[0, 0], x0[0, 1], x0[0, 2], "r")
    plt.title(
        type(model).__name__ + " Prediction of Lorenz Attractor, rho=" + str(rho)
    )
    ax.view_init(18, -113)
    plt.show()

    fig = plt.figure()

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(t, x_t[:, 0], linewidth=1)
    ax.plot(t, ynn[:, 0], "g.")
    plt.title(type(model).__name__ + " Prediction of Lorenz Attractor, rho= " + str(rho))
    ax.set_ylabel("x(t)")

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(t, x_t[:, 1], linewidth=1)
    ax.plot(t, ynn[:, 1], "g.")
    ax.set_ylabel("y(t)")

    ax = fig.add_subplot(3, 1, 3)
    ax.plot(t, x_t[:, 2], linewidth=1)
    ax.plot(t, ynn[:, 2], "g.")
    ax.set_ylabel("z(t)")
    ax.set_xlabel("t")
    plt.show()
```
We test the LSTM model.

```python
np.random.seed(133) ## Set once before multiple calls to random() function
LSTM_predict(model, rho = 17)
LSTM_predict(model, rho = 35)
```

**RNN**

Similarly to the LSTM, we create an RNN model.

```python
class LorenzRNN(nn.Module):
    def __init__(self):
        super(LorenzRNN, self).__init__()
        self.rnn = nn.RNN(input_size=3, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        _, h = self.rnn(x)
        x = self.fc(h[-1])
        return x
```

We train the RNN.

```python
# Train the neural network using all rho_train values
model = LorenzRNN()
model = train_NN(model, input_train=nn_input_train, output_train=nn_output_train, num_epochs=100)
```
And test it.

```python
# Now see how well the NN works for future state prediction 
np.random.seed(133) ## Set once before multiple calls to random() function
LSTM_predict(model, rho = 17)
LSTM_predict(model, rho = 35)
```

**Echo State Network**

We use the Python library easyesn, developed by Roland Zimmermann and Luca Thiede. https://github.com/kalekiu/easyesn/blob/master/examples/PredictionExample.ipynb

First, we install easyesn using `!pip install easyesn`. We then import the necessary packages.

```python
from easyesn.optimizers import GradientOptimizer
from easyesn import PredictionESN
from easyesn.optimizers import GridSearchOptimizer
from easyesn import helper as hlp
```

Create and train the model.

```python
esn = PredictionESN(n_input=1, n_output=1, n_reservoir=50, leakingRate=0.2, regressionParameters=[1e-2], solver="lsqr", feedback=False)
esn.fit(nn_input_train, nn_output_train, transientTime="Auto", verbose=1)
```

Then test it.

```python
rho = 17
x0 = -15 + 20 * np.random.random((2, 3))
x_t = integrate.odeint(lorenz_deriv, x0[0, :], t, args=(rho, sigma, beta))

ynn = np.zeros((len(t), 3))
ynn[0, :] = x0[0, :]
print("x0     :", ynn[0, :])
with torch.no_grad():
    y0 = torch.tensor(ynn[0, :], dtype=torch.float32)
    for j in range(len(t) - 1):
        y1 = esn.predict(y0) 
        y0 = y1
        ynn[j + 1, :] = y1.squeeze()
        if (j + 1) % 100 == 0:
            print("predict:", ynn[j + 1, :])
            print("actual :", x_t[j, :])
```

## 4. Computational Results

### i. Forcasting with a Neural Network

![image](https://github.com/marybun/machine_learning/assets/108769794/d36b3e87-599a-4f9d-970d-cd6b0075c702)

![image](https://github.com/marybun/machine_learning/assets/108769794/b4a9f18e-8039-4bb9-8516-24faacb879ad)

![image](https://github.com/marybun/machine_learning/assets/108769794/37ac8b90-50e0-46b3-8511-e1fae75c49d2)

![image](https://github.com/marybun/machine_learning/assets/108769794/95d08374-afe8-4eee-b1c4-bfb019ac9e19)

For all the following graphs, the blue solid line represents the true values, while the green dotted line represents the predicted values.

![image](https://github.com/marybun/machine_learning/assets/108769794/a9fc3482-06ab-4059-b466-66fd812ff429)

![image](https://github.com/marybun/machine_learning/assets/108769794/e7cc89bf-9f8c-4c1b-9f86-8375105e62d0)

![image](https://github.com/marybun/machine_learning/assets/108769794/0d1dafda-4cc2-46e7-ba45-b950dbfa5a43)

![image](https://github.com/marybun/machine_learning/assets/108769794/5ac24634-54cd-435f-b8cf-d98f829ad068)

### ii. Comparing with Other Neural Networks

**LSTM**

![image](https://github.com/marybun/machine_learning/assets/108769794/b0ea4a0f-aeb4-414e-b2bc-136fe0bed7e0)

![image](https://github.com/marybun/machine_learning/assets/108769794/17b79a28-d34d-4495-a736-ba22eca4984c)

![image](https://github.com/marybun/machine_learning/assets/108769794/a3fc6fcb-d0c5-485c-a8eb-1fbc41698e91)

![image](https://github.com/marybun/machine_learning/assets/108769794/41bd6c8f-a6a3-4abf-86c3-b29e7ea2eddb)

![image](https://github.com/marybun/machine_learning/assets/108769794/b67ce2dd-3414-40ad-9399-ea9821212114)

**RNN**

![image](https://github.com/marybun/machine_learning/assets/108769794/c32713a2-1cbd-4ba3-a486-f1c4331dd1d3)

![image](https://github.com/marybun/machine_learning/assets/108769794/6dbda497-8eb3-402e-8ed2-1d9343d82fc5)

![image](https://github.com/marybun/machine_learning/assets/108769794/dd6d5723-5af3-4a18-afa8-27a65f211ed8)

![image](https://github.com/marybun/machine_learning/assets/108769794/1ccb154f-a6c7-47e1-9255-31653f38359f)

![image](https://github.com/marybun/machine_learning/assets/108769794/a62c9e2b-00ad-4476-9a0a-7d25e4534540)

**Echo State Network**

![image](https://github.com/marybun/machine_learning/assets/108769794/d97e268c-b650-4887-ae8e-76f8924f4d25)

![image](https://github.com/marybun/machine_learning/assets/108769794/98358cee-5ad0-4736-94be-b578fe2c1943)

![image](https://github.com/marybun/machine_learning/assets/108769794/7acffe95-eb0c-4fc3-a541-920372b56d07)

![image](https://github.com/marybun/machine_learning/assets/108769794/92c2b8ec-f318-4f00-bb03-fadd7d2aba97)

## 5. Summary and Conclusions

### i. Forcasting with a Neural Network

The feed forward network performs the best. Looking at the Lorenz attractor, we can see the true and predicted curves drifting away from each other, however, they both tend to make a spiraling shape, converging to some point.

### ii. Comparing with Other Neural Networks

The RSTM and RNN do poorly, with the RNN taking an especially long time to train. These models make no effort to minimize the error.

The Echo State Network, on the other hand, trains very quickly (11 seconds). The results are inconsistent in each run, but sometimes result in a spiral-like shape that we would expect.
