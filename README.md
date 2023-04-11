# Curve Fitting
Author: Mary Bun

Table of Contents
1. Introduction and Overview
2. Theoretical Background
3. Algorithm Implementation and Development 
    1. Fitting a Model
    2. Error Landscape
    3. Training and Testing Data
4. Computational Results
5. Summary and Conclusions

[curve_fit.ipynb](https://github.com/marybun/machine_learning/blob/main/curve_fit.ipynb) provides basic examples of curve fitting in Python, following HW1 in EE 399A, Introduction to Machine Learning for Science and Engineering, a UW course designed by J. Nathan Kutz for Spring 2023.
## 1. Introduction and Overview

The purpose of this notebook is to become familar with curve fitting in regression. Curve fitting is used to fit data to a curve that best represents the underlying pattern of the data, which can then be used to predict other data points. This notebook will explore the basics of curve fitting in Python, such as how to fit a defined model, making initial guesses, how to fit polynomial models, and overfitting.
## 2. Theoretical Background

**Least-Squares** is a commonly used method of finding a line or curve of best-fit. It minimizes the sum of the squared differences between the predicted values and the true values. In this notebook, we will use this method to minimize the Root Mean Squared Error (RMSE) between the data and various models.

**Root Mean Squared Error (RMSE)** is a measure of how accurate a model is to the data. It is calculated by taking the square root of the mean of the squared error (differences) between the predicted values and the true values. The lower the RMSE, the more accurate the model is.

**fmin** is a function in the scipy.optimize module that finds a minimum of a given function when given the function and an initial guess of the coefficients of that function. fmin gives the minimum closest to the initial guess, so it is important to make an initial guess that is close to the global minimum. In our case, the function to be minimized is the RMSE between the predicted and true data. fmin will return the coefficients for the model that will produce the minimum.

**polyfit** creates a polynomial model given the degree and training data.

**polyval** returns the predicted data given the polynomial model and the X values.
## 3. Algorithm Implementation and Development 
### 1. Fitting a Model
We fit 31 given data points to the model given by the equation: $f(x) = A\cos(Bx) + Cx + D$

And calculate the RMSE value using the equation described in Section 2: $E = \sqrt{\frac{1}{n}\Sigma_{j=1}^{n}(f(x_j)-y_j)^2}$, where $n$ is the number of data points, $f(x_j)$ are the predicted values (can also be written as $\hat{y_j}$), and $y_j$ are the true values.
We then make an initial guess $A0$, which will be used to find the coefficients $A, B, C,$ and $D$ to produce a minimum RMSE.
### 2. Error Landscape
We can use different coefficients to produce a more accurate model. To do this, we will sweep through the coefficients two at a time, and find which values produce minimum errors.
### 3. Training and Testing Data
To further examine the accuracy of models to our data, we will use models of varying degrees and see how well they fit with the original data. We will do this by using the first 20 original data points as training data, and using the rest as test data. polyfit will create a polynomial model that fits the training data. A good model should follow all the data with little error.
The 19th degree polynomial model leads to overfitting, which we can bypass by training the models on the outer data points and testing the middle points. We train the models on the first 10 and last 11 data points, and use the middle 10 points as testing data.
## 4. Computational Results

![image](https://user-images.githubusercontent.com/108769794/231020268-774e7a5b-5216-45d4-8e44-05d378a09c98.png)

As seen above, the fitted data follows the trend of the original data, but could be more accurate. Sweeping the coefficients and plotting color landscapes can help us find where the minima are and especially which coefficients produce the global minimum.

![image](https://user-images.githubusercontent.com/108769794/231073659-d4412fc6-3649-4a92-ab73-53cd445ae65f.png)
![image](https://user-images.githubusercontent.com/108769794/231073682-9438661c-d426-4852-ba37-acd111e3c5ce.png)
![image](https://user-images.githubusercontent.com/108769794/231073703-0fe5916d-9bf9-4cbe-81b0-2e872633e2f4.png)
![image](https://user-images.githubusercontent.com/108769794/231073727-9c8e3cda-0c53-4f99-b5a8-2720411ed794.png)
![image](https://user-images.githubusercontent.com/108769794/231073739-6f491d6a-0d76-4222-9956-b0ab0f1c2e4f.png)
![image](https://user-images.githubusercontent.com/108769794/231073758-773a7286-9fd6-488f-b146-a9d41b80d3bc.png)


The color maps above display the RMSE values for swept coefficients. The darker colors indicate a smaller RMSE value. Overall, we observe that the coefficient $C$ has little effect on the RMSE values. $A$ is more sensitive than $B$, and $A$ and $D$ are similarly sensitive. 

Using three polynomial models of degrees 1, 2, and 19, and using the first 10 data points for training and the rest for testing, we produce the following graphs.

![image](https://user-images.githubusercontent.com/108769794/231020572-09fe5ab3-ac24-4d23-8a6e-543ef9dfd370.png)
![image](https://user-images.githubusercontent.com/108769794/231020582-78cc0685-6390-458a-b3c1-3d8e66606206.png)
![image](https://user-images.githubusercontent.com/108769794/231020598-8e5d4449-d540-48d5-a38d-56f3ba9b8493.png)

One might assume increasing the degree of the polynomial will give a more accurate model, since more coefficients allow for more flexibility, but with the 19th degree polynomial above, this is not the case. This phenomenon is called **overfitting**. It occurs when a model is too complex and fits the noise of the data, rather than the pattern.
We also observe that the 2nd and 19th degree polynomial models deviate more from the data as X increases, because they weren't trained on the later points.

If we instead use the first 10 and last 11 points as training data, and the 10 in between as testing data, we get the following graphs.

![image](https://user-images.githubusercontent.com/108769794/231020765-967bb324-2fc6-4b2d-8b48-dda42cf8547e.png)
![image](https://user-images.githubusercontent.com/108769794/231020779-02afb9b5-324d-4ff9-9f00-4082dcb23e2f.png)
![image](https://user-images.githubusercontent.com/108769794/231020787-892379eb-a1ae-4b51-baac-3ad559b92498.png)

We can see that the 2nd degree polynomial fits the trend of the data much more accurately. Additionally, the 1st degree and 2nd degree polynomials are very similar, which implies that a linear line of best-fit is more accurate than a parabolic curve. The 19th degree polynomial has a large increase around X = 15, which is caused by overfitting.
## 5. Summary and Conclusions

Curve fitting is an essential part of data analysis and provides many insights on a dataset, such as the relationships between variables and predictions for future data points. They are also useful in visualizing data, which grants a better understanding of the aforementioned attributes.

