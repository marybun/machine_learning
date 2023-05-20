# <a name="SHRED"></a>SHallow REcurrent Decoder (SHRED)

Table of Contents
1. Introduction and Overview
2. Theoretical Background
3. Algorithm Implementation and Development 
    1. SHallow REcurrent Decoder (SHRED)
    2. Performance over Time Lag
    3. Performance over Noise
    4. Performance over Number of Sensors
4. Computational Results
5. Summary and Conclusions

## Abstract

[SHRED.ipynb](https://github.com/marybun/machine_learning/blob/main/HW6/SHRED.ipynb) performs analyses on how well the SHallow REcurrent Decoder (SHRED) predicts a dataset of weekly mean sea-surface temperature from 1992 to 2019. We analyse the model's performance over a range of time lags, noise values, and sensors.

## 1. Introduction and Overview

This notebook uses the SHallow REcurrent Decoder (SHRED) model, as implemented in the paper, "Sensing with Shallow Recurrent Decoder Networks" by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz. The code and data of which can be found in the pyshred repository (https://github.com/shervinsahba/pyshred). In section i, we implement the SHRED model and see how its predictions compare to the ground truth. In section ii, we investigate how time lag influences the model's performance. In section iii, we create Gaussian noise, scaling it with $\alpha$, and again, seeing how it influences the model's performance. Finally, in section iv, we see how the model performs with varying numbers of sensors.

## 2. Theoretical Background

In measuring the full-state of a system, it is often impractical or impossible to make many discrete measurements. One solution for this is to use sensors to predict future outcomes. However, to account for noise and physical limitations, using sensors in real-life applications requires meticulous location placement to achieve an accurate result. Systems may also require a large amount of sensors to fully capture their state.

**SHallow REcurrent Decoder (SHRED)** is a neural network structure that contains an LSTM and a shallow decoder. Using sensor trajectories and past information, compared to static sensor measurements, SHRED makes more accurate predictions with far fewer sensors, which can also be randomly placed.

**Recurrent Neural Network (RNN)** is a type of neural network used for processing sequential data. It keeps a memory of previous information to make predictions.

**Long Short-Term Memory (LSTM)** is a type of RNN that uses gates to control the flow of information. These gates allow the network to remember and forget specific information.

## 3. Algorithm Implementation and Development

### i. SHallow REcurrent Decoder (SHRED)

First, we import the necessary packages and set the number of epochs to 200. Increasing the number of epochs will improve the performance of the model, but take more computation time.

```python
import numpy as np
from processdata import load_data, TimeSeriesDataset
import models
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from processdata import load_full_SST

load_X = load_data('SST')

num_epochs = 200
```

Next, we define the SHRED performance function that creates, trains, and tests the SHRED model and plots the results. By default, the number of sensors is 3, randomly placed, and the trajectory length (lags) is 52, which is a full year of measurements.

```python
# Function to train and valid the model for a specific lags and num_sensors
# Return mean square error as comparing to the ground truth values.
def SHREDperformance(lags=52, num_sensors=3, alpha=0, isPlot=False, num_epochs=50):
    n = load_X.shape[0]
    m = load_X.shape[1]
    # Randomly select num_sensors sensor locations and set the trajectory length (lags) to lags, 
    # corresponding to one year of measurements.
    sensor_locations = np.random.choice(m, size=num_sensors, replace=False)

    # select indices to divide the data into training, validation, and test sets.
    train_indices = np.random.choice(n - lags, size=1000, replace=False)
    mask = np.ones(n - lags)
    mask[train_indices] = 0
    valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
    valid_indices = valid_test_indices[::2]
    test_indices = valid_test_indices[1::2]

    # sklearn's MinMaxScaler is used to preprocess the data for training and we generate input/output pairs for the training, validation, and test sets.

    sc = MinMaxScaler()
    sc = sc.fit(load_X[train_indices])
    transformed_X = sc.transform(load_X)

    ### Generate input sequences to a SHRED model
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(len(all_data_in)):
        all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

    ### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
    ## Add Gaussian noise with zero mean if alpha is not zero
    if alpha != 0.0:
        avg_x = torch.mean(abs(train_data_in))
        std_dev = alpha * train_data_in
        noise = torch.randn_like(train_data_in) * std_dev 
        train_data_in = train_data_in + noise
    
    valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
    test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

    ### -1 to have output be at the same time as final sensor measurements
    train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
    valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
    test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

    train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
    valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
    test_dataset = TimeSeriesDataset(test_data_in, test_data_out)

    # We train the model using the training and validation datasets.
    shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
    validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=num_epochs, lr=1e-3, verbose=True, patience=5)

    # Finally, we generate reconstructions from the test set and print mean square error compared to the ground truth.

    test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
    test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
    mse=np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth)
    print("mean square error:", mse)

    # Plot if asked for
    if(isPlot):
        # SST data with world map indices for plotting
        full_SST, sst_locs = load_full_SST()
        full_test_truth = full_SST[test_indices, :]

        # replacing SST data with our reconstruction
        full_test_recon = full_test_truth.copy()
        full_test_recon[:,sst_locs] = test_recons

        # reshaping to 2d frames
        for x in [full_test_truth, full_test_recon]:
            x.resize(len(x),180,360)
        plotdata = [full_test_truth, full_test_recon]
        labels = ['truth','recon']
        fig, ax = plt.subplots(1,2,constrained_layout=True,sharey=True)
        for axis,p,label in zip(ax, plotdata, labels):
            axis.imshow(p[0])
            axis.set_aspect('equal')
            axis.text(0.1,0.1,label,color='w',transform=axis.transAxes)
    return mse
```

### ii. Performance over Time Lag

Using a `for` loop to iterate through lag values, we plot the MSE of the model over lag.

```python
# Do an analysis of the performance as a function of the time lag variable
num_sensors = 3
alpha = 0.0 # no noise added
# Loop through a range of lags
lags = range(4,53,4) 
mses_by_lags = np.zeros(len(lags))
for i in range(len(lags)):
    mses_by_lags[i] = SHREDperformance(lags[i], num_sensors, alpha, isPlot=False, num_epochs=num_epochs)
plt.plot(lags, mses_by_lags)
plt.ylabel("mean square error")
plt.xlabel("lags")
plt.title("Performance of SHRED as a function of the time lag variable")
```
### iii. Performance over Noise

We iterate through a range of alpha values, which influences the amount of noise added to the data. We plot the MSE over alpha.

```python
# Do an analysis of the performance as a function of noise (add Gaussian noise to data)
# Noise has mean zero and standard deviation equal to alpha times the mean absolute value of the field
lags = 52
num_sensors = 3
# Loop through a range of noises
alphas = np.arange(0.0,0.21,0.02) 
mses_by_alpha = np.zeros(len(alphas))
for i in range(len(alphas)):
    mses_by_alpha[i] = SHREDperformance(lags, num_sensors, alpha=alphas[i], isPlot=False, num_epochs=num_epochs)
plt.plot(alphas, mses_by_alpha)
plt.ylabel("mean square error")
plt.xlabel("noise (alpha)")
plt.title("Performance of SHRED as a function of noise (alpha)")
```

### iv. Performance over Number of Sensors

We iterate through a number of sensors, then plot the MSE over number of sensors.

```python
# Do an analysis of the performance as a function of the number of sensors
lags = 52
alpha = 0.0 # no noise added
# Loop through a range of number of sensors
num_sensors = range(3,51,4) 
mses_by_sensors = np.zeros(len(num_sensors))
for i in range(len(num_sensors)):
    mses_by_sensors[i] = SHREDperformance(lags, num_sensors[i], alpha, isPlot=False, num_epochs=num_epochs)
plt.plot(num_sensors, mses_by_sensors)
plt.ylabel("mean square error")
plt.xlabel("number of sensors")
plt.title("Performance of SHRED as a function of the number of sensors")
```

## 4. Computational Results

### i. SHallow REcurrent Decoder (SHRED)

mean square error: 0.031557165

![image](https://github.com/marybun/machine_learning/assets/108769794/e67c5daa-34c4-43bd-9753-26ae38f0f7ef)

### ii. Performance over Time Lag

![image](https://github.com/marybun/machine_learning/assets/108769794/c892e01b-6606-4606-bfff-9629ac1763ce)

### iii. Performance over Noise

![image](https://github.com/marybun/machine_learning/assets/108769794/301a529d-57c7-4099-a120-d0774137e37f)


### iv. Performance over Number of Sensors

![image](https://github.com/marybun/machine_learning/assets/108769794/72d54294-1999-418b-a49e-70fe78b11c47)

## 5. Summary and Conclusions

### i. SHallow REcurrent Decoder (SHRED)

Using only 3 sensors, SHRED accurately predicts the sea surface temperature across the globe.

### ii. Performance over Time Lag

When the time lag is increased, the model is given much more information in terms of trajectory length, meaning it can perform much better.

### iii. Performance over Noise

As expected, increasing the noise increases the error, however, the model still does relatively well even at the highest $\alpha$ value of 0.2.

### iv. Performance over Number of Sensors

Similarly to the time lag analysis, increasing the number of sensors provides more information for the model, thus improving performance significantly. There is an especially significant drop in error between 3 and 10 sensors, which shows the importance of finding the ideal number of sensors, such that you have a small amount of sensors while still having enough to get an accurate prediction.
