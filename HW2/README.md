# <a name="correlation"></a>Correlation

Table of Contents
1. Introduction and Overview
2. Theoretical Background
3. Algorithm Implementation and Development 
    1. Correlation Matrix
    2. Correlated Images
    3. 10x10 Correlation Matrix
    4. Singular Value Decomposition
4. Computational Results
5. Summary and Conclusions

## Abstract

[correlation.ipynb](https://github.com/marybun/machine_learning/blob/main/HW2/correlation.ipynb) explores the concept of correlation and introduces singular value decomposition, following Homework 2. Using [yalefaces.mat](https://github.com/marybun/machine_learning/blob/main/HW2/yalefaces.mat), we will find the correlation between images and find the most prominent aspects of each face using eigenvectors, eigenvalues, and singular value decomposition.

## 1. Introduction and Overview

This notebook uses the MATLAB file [yalefaces.mat](https://github.com/marybun/machine_learning/blob/main/HW2/yalefaces.mat), which is a matrix (which we will name $X$) containing 39 faces with 65 lighting conditions (2414 faces in total). Python indexing starts at 0, so throughout the report, **images will be refered to with their index numbers**. Image 0 is the first image, Image 1 is the second, and so on. Each face is a 32x32 grayscale pixel image, which is represented as a column vector of size 1024. The size of the matrix is 1024x2414. We will find the correlation matrix of the first 100 images to find which 2 images of those are the most and least correlated. We will compute a 10x10 correlation matrix using images from the original matrix $X$, and calculate the eigenvectors and eigenvalues of $X$. We will compare these eigenvectors to singular value decomposition (SVD) modes and use SVD to find the most important aspects of the images of $X$.

## 2. Theoretical Background

**Correlation** refers to the strength of a relationship between two variables, in this case, images. The correlation matrix is calculated with the equation $\bf{c_{jk}} = \bf{x_j^T x_k}$.

**Eigenvalues and Eigenvectors** are linear algebra concepts that describe the most important parts of a square matrix. A square matrix that applies a linear transformation can be described by eigenvectors, which are the directions of the transformation, and their corresponding eigenvalues are the magnitudes. $\bf{Av} = \lambda \bf{v}$, where $\lambda$ is the eigenvalue and $\bf{v}$ is the eigenvector.

**Singular Value Decomposition (SVD)** takes a matrix and decomposes it into 3 matrices, following the form $\bf{M} = \bf{U \Sigma V^\*}$. The matrix $\bf{U}$ has columns orthogonal to each other, and $\bf{V^\*}$ is the conjugate transpose of a matrix whose columns are also orthogonal to each other. These columns are known as singular vectors and are essentially directions that represent the strongest patterns in a dataset. $\bf{\Sigma}$ is a diagonal matrix of singular values, which are sorted in descending order and represent the magnitudes of the singular vectors. While eigenvalues and eigenvectors are similar to SVD, they can only be used for square matrices, while SVD can be used for any rectangular matrix.

## 3. Algorithm Implementation and Development

We load the yalefaces.mat file into Python by running the following:

```python
from scipy.io import loadmat
results=loadmat('yalefaces.mat')
X=results['X']
```

### i. Correlation Matrix

We compute an 100x100 correlation matrix of the first 100 images in yalefaces.mat and name it $C$.

```python
C = np.zeros((100, 100))

for j in range(100):
    for k in range(100):
        C[j, k] = X[:, j].T @ X[:, k]
```

We can then plot it.

```python
plt.figure(figsize=(7,6))
plt.pcolor(C)
plt.title('Correlation Matrix of yalefaces.mat')
plt.xlabel('Image k')
plt.ylabel('Image j')
plt.colorbar()
plt.show()
```
### ii. Correlated Images

Using the correlation matrix $C$, we can find the two greatest and two least correlated images. Since we don't want the correlations between an image and itself and we don't want any repeats of correlations, we take the upper triangle of $C$.

```python
# We only need one triangle of the matrix to get the correlations between every image
C_triu = np.triu(C, 1)

# Find the indicies for the most correlated images
C_maxcorr_idx = np.unravel_index(np.argmax(C_triu), C.shape)

# Set the values of the diagonal and below to 1000 so np.min finds the min within the upper triangle
C_triu += 1000*np.tril(np.ones((100, 100)))
# Find the indicies for the least correlated images
C_mincorr_idx = np.unravel_index(np.argmin(C_triu), C.shape)
```
To plot the images, we need to reshape them to 32x32 matrices, since each image was originally a column vector.

```python
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.imshow(X[:, C_maxcorr_idx[0]].reshape(32,32))
plt.title('Image ' + str(C_maxcorr_idx[0]))
plt.subplot(122)
plt.imshow(X[:, C_maxcorr_idx[1]].reshape(32,32))
plt.title('Image ' + str(C_maxcorr_idx[1]))
plt.suptitle('Most Correlated Images')
plt.show()

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.imshow(X[:, C_mincorr_idx[0]].reshape(32,32))
plt.title('Image ' + str(C_mincorr_idx[0]))
plt.subplot(122)
plt.imshow(X[:, C_mincorr_idx[1]].reshape(32,32))
plt.title('Image ' + str(C_mincorr_idx[1]))
plt.suptitle('Least Correlated Images')
plt.show()
```
### iii. 10x10 Correlation Matrix

We can compute a different correlation matrix, $C_10$, which will be the correlations between the 0, 312, 511, 4, 2399, 112, 1023, 86, 313, and 2004th images.

```python
X_10 = X[:, [0, 312, 511, 4, 2399, 112, 1023, 86, 313, 2004]]
C_10 = np.zeros((10, 10))

for j in range(10):
    for k in range(10):
        C_10[j, k] = X_10[:, j].T @ X_10[:, k]
```
We then plot it.

```python
plt.figure(figsize=(6,5))
plt.pcolor(C_10)
plt.xlabel('Image k')
plt.ylabel('Image j')
plt.title('Correlation Matrix of 10 Images')
plt.colorbar()
plt.show()
```

We create the matrix $Y = XX^T$ and compute the six eigenvectors corresponding to the largest eigenvalues using numpy.linalg.eig.

```python
# Y = XX^T
Y = X @ X.T

# Find the first six eigenvectors with the largest eigenvalues
w, v = la.eig(Y)
np.shape(v[:, 0:6])
print(v[:, 0:6])
```
### iv. Singular Value Decomposition

We can use SVD to find the first six principle component directions of original matrix $X$.
```python
u, s, vh = la.svd(X)
print(u[:, 0:6])
```

We can compare the first eigenvector to the first SVD mode by taking the norm of the differences of the absolute values of each.

```python
la.norm(abs(v[:, 0]) - abs(u[:, 0]))
```
We calculate the percentage of the variance of the first 6 SVD modes.

```python
for mode in range(6):
    print('Variance for mode', str(mode), '=', 100 * (s[mode] / np.sum(s)), '%')
```
And plot them by reshaping them to 32x32.

```python
plt.figure(figsize=(20, 10))
for i in np.arange(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(u[:, i].reshape(32,32))
    plt.title('SVD mode ' + str(i))
```

## 4. Computational Results

### i. Correlation Matrix

![image](https://user-images.githubusercontent.com/108769794/232979251-d504e71a-439c-4b87-84f7-1aedefc927ed.png)

### ii. Correlated Images

![image](https://user-images.githubusercontent.com/108769794/232979316-f903f00f-27d5-4cbc-b6e9-97e994856895.png)

![image](https://user-images.githubusercontent.com/108769794/232979394-b728a3b4-01db-49f3-9143-3635e4ccfac2.png)

### iii. 10x10 Correlation Matrix

![image](https://user-images.githubusercontent.com/108769794/232979512-8095117e-a2f1-4050-b08a-ab75017ca5f6.png)

### iv. Singular Value Decomposition

Norm of difference between absolute values of 1st eigenvector and 1st SVD mode: 7.51892072605625e-16

Variance for mode 0 = 16.614046686527438%

Variance for mode 1 = 7.605298911138135%

Variance for mode 2 = 3.1168860138255963%

Variance for mode 3 = 2.6657683644942196%

Variance for mode 4 = 1.5555497307794566%

Variance for mode 5 = 1.4974371809143299%

![image](https://user-images.githubusercontent.com/108769794/232979893-71700842-ce1e-4d12-b190-4216ea7084f0.png)

## 5. Summary and Conclusions

### i. Correlation Matrix

The correlation matrix is symmetrical across the line $y=x$, since we are calculating the correlation between the same two images across that line. The warmer the color, the higher the correlation.

### ii. Correlated Images

The most correlated images, Image 86 and Image 88, look very similar, which explains their high correlation. The least correlated images, Image 54 and Image 64, are very different, with 54 having a visible face, and 64 being almost completely dark.

### iii. 10x10 Correlation Matrix

Similar to the correlation matrix $C$, $C_10$ displays the correlation between each image, with the same symmetry across $y=x$. 

### iv. Singular Value Decomposition

Comparing the 1st eigenvector with the 1st SVD mode, we can see they are essentially the same, with the difference being almost 0.

The variances for each SVD mode is in decreasing order. SVD mode 0 has the highest variance because it explains the data the most.

Graphing the modes out, we can see that each one looks like a face, with each mode highlighting the most important aspects of a face, such as the eyes, nose, and mouth. They also show the importance of lighting, with each mode having different lighting scenes that were common with faces in the original matrix $X$.
