# <a name="correlation"></a>Correlation

Table of Contents
1. Introduction and Overview
2. Theoretical Background
3. Algorithm Implementation and Development 
    1. Correlation Matrix
    2. Correlated Images
    3. Singular Value Decomposition
4. Computational Results
5. Summary and Conclusions

[correlation.ipynb](https://github.com/marybun/machine_learning/blob/main/HW2/correlation.ipynb) explores the concept of correlation and introduces singular value decomposition, following Homework 2

## 1. Introduction and Overview

This notebook uses the MATLAB file [yalefaces.mat](https://github.com/marybun/machine_learning/blob/main/HW2/yalefaces.mat), which is a matrix containing 39 faces with 65 lighting conditions (2414 faces in total). Each face is a 32x32 grayscale pixel image, which is represented as a column vector of size 1024. The size of the matrix is 1024x2414.

## 2. Theoretical Background

**Correlation** refers to the strength of a relationship between two or more constructs, in this case, images. The correlation matrix is calculated with the equation $\bf{c_{jk}} = \bf{x_j^T x_k}$

**Singular Value Decomposition (SVD)**
