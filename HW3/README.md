# <a name="classify"></a>Classifying a Dataset

Table of Contents
1. Introduction and Overview
2. Theoretical Background
3. Algorithm Implementation and Development 
    1. Truncated SVD of MNIST
    2. Building A Classifier
    3. Classifying All Digit Pairs
    4. Classifying All Digits
    5. Comparing Classifier Models
4. Computational Results
5. Summary and Conclusions

[classify.ipynb](https://github.com/marybun/machine_learning/blob/main/HW3/classify.ipynb) showcases different methods to classifying the digits in the MNIST dataset, following Homework 3. Using SVDs, we find which of the three methods (LDA, SVM, or Decision Trees) classifies the data most accurately.

## 1. Introduction and Overview

This notebook uses the MNIST dataset, which is a dataframe of size 70000x784. Each row is a vectorized 28x28 pixel image of a handwritten digit (0 to 9). We start by taking the full SVD of the data, then finding the number of modes necessary for image reconstruction to cut down on computation time. We use this to take the truncated SVD, which we use for 3 classification methods: Linear Discriminant Analysis, Support Vector Machines, and Decision Trees.

## 2. Theoretical Background

**Singular Value Decomposition (SVD)** decomposes a matrix into 3 matrices, following the form $\bf{M} = \bf{U \Sigma V^\*}$. The matrix $\bf{U}$ has columns orthogonal to each other, and $\bf{V^\*}$ is the conjugate transpose of a matrix whose columns are also orthogonal to each other. These columns are known as singular vectors and are essentially directions that represent the strongest patterns in a dataset. $\bf{\Sigma}$ is a diagonal matrix of singular values, which are sorted in descending order and represent the magnitudes of the singular vectors.

**Linear Discriminant Analysis (LDA)** is a classifier model that discriminates between classes by taking linear combinations of the most important features of an image, called discriminant functions, and finding which images match which discriminant functions.

**Cross-Validation** is a technique to evaluate the performance of a model by taking random training and testing points and running the model on them. This process is repeated to get a sense of how well the model does on different data points. It also helps reduce the risk of overfitting.

**Support Vector Machine (SVM)** is classifier model that works by mapping the data onto a higher dimensional plane, and classifying by finding the hyperplane that separates the classes the most.

**Descision Tree** is a classifier model that classifies data in a heirarchical fashion, by recursively splitting the data. It continues to split the data until they are all classified correctly.

## 3. Algorithm Implementation and Development

### i. Truncated SVD of MNIST

First, we load the MNIST dataset and relevant packages.
```python
import numpy as np
from numpy import linalg as la
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the MNIST data
mnist = fetch_openml('mnist_784', parser='auto')
X = mnist.data / 255.0  # Scale the data to [0, 1]
y = mnist.target
```

The data is a pandas dataframe of size 70000x784 (images by pixels). We can plot the images by taking the transpose, converting to a NumPy array, and reshaping to 28x28.
```python
# Plot the first 9 images of MNIST in grayscale
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(np.array(X.T)[:, i].reshape(28,28), cmap='gray')
    plt.title('Image ' + str(i))
plt.tight_layout()
```
MNIST is a very large dataset and requires a large amount of memory to use the np.linalg.svd function, so we instead use TruncatedSVD.

```python
# Compute the full SVD of MNIST
n_components = 784 # The maximum amount of components
svd = TruncatedSVD(n_components=n_components)  
X_svd = svd.fit_transform(X.T)

U = X_svd / svd.singular_values_
S = np.diag(svd.singular_values_)
VT = svd.components_
```

We can plot the first 9 U modes to see the most prominent features of the data.

```python
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(U[:, i].reshape(28,28), cmap='gray')
    plt.title('Image ' + str(i))
plt.suptitle('U modes')
plt.tight_layout()
```

We can plot the singular value spectrum by plotting each value of the S matrix.

```python
plt.figure(figsize=(10,6))
plt.plot(svd.singular_values_,'o', markersize=5)
plt.xlabel('Singular value index')
plt.ylabel('Singular value')
plt.title('MNIST Singular Value Spectrum')
plt.grid(True)
plt.show()
```

To find how many modes are necessary for good image reconstruction, we take the cumulative sum of the singular values when it reaches 90% of the total sum of singular values.

```python
total_sum = np.sum(svd.singular_values_)
cumulative_sum = np.cumsum(svd.singular_values_)/total_sum   # Scaled to [0, 1]
plt.figure(figsize=(10,6))
plt.plot(cumulative_sum,'o', markersize=5)
plt.xlabel('Singular value index')
plt.ylabel('Singular value')
plt.title('Cumulative Singular Values')
num_modes = np.where(cumulative_sum >= 0.9)[0][0] + 1
print('Number of modes necessary for good image reconstruction:', num_modes)

plt.grid(True)
plt.show()
```

We can also compute the rank of the dataset to find the amount of modes that retains even more information using
`la.matrix_rank(X.T)`.

Using the number of modes that constitutes 90% of the data, we take the truncated SVD of the data.

```python
n_components = 344 # Number of modes necessary for good image reconstruction
svd = TruncatedSVD(n_components=n_components)  # Compute the first 'n_components' singular vectors and values
Xf_svd = svd.fit_transform(X.T)

Uf = Xf_svd / svd.singular_values_
Sf = np.diag(svd.singular_values_)
VfT = svd.components_
```

And plot the first 9 images of the truncated matrix $U_f \Sigma_f V^\*$.

```python
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(np.array(Uf@Sf@VfT)[:, i].reshape(28,28), cmap='gray')
    plt.title('Image ' + str(i))
plt.suptitle('Truncated Matrix')
plt.tight_layout()
```

We can plot the 10% of data lost by subtracting the original matrix by the truncated matrix.

```python
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow((np.array(X.T)[:, i] - np.array(Uf@Sf@VfT)[:, i]).reshape(28,28), cmap='gray')
    plt.title('Image ' + str(i))
plt.suptitle('Loss Matrix')
plt.tight_layout()
```
In the SVD matrix, the left and right singular vectors $\bf{U}$ and $\bf{V^\*}$ represent the modes of the images, that is, the most prominent features of the images. The matrix $\bf{\Sigma}$ is the diagonal matrix of singular values that scale each singular vector. The higher the magnitude of the singular value, the more prominent the feature is.

Next, we project 3 V modes (1, 2, and 4) onto a 3D plot with colored digit labels.

```python
plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
colors = ['black','red','green','blue','purple','yellow','cyan','magenta','gray','brown']
for i in np.arange(10):
    idx = np.where(y == str(i))[0][0:99]
    c = colors[i]
    plt.plot(VfT.T[idx, 1],VfT.T[idx, 2],VfT.T[idx,4],'.',color=c,markeredgecolor='k', alpha=0.4,ms=12, label=('Digit '+str(i)))
ax.set_xlabel('Vf1')
ax.set_ylabel('Vf2')
ax.set_zlabel('Vf4')
plt.title('3 V Modes with Colored Digit Labels')
plt.legend()
```

### ii. Building A Classifier

We can classify a pair of digits, in this case, 0 and 1, by concatenating the indices of the labels corresponding to those digits.

```python
indexTo0 = np.where(y == '0')[0]
indexTo1 = np.where(y == '1')[0]
print("Number of rows for digit '0':", len(indexTo0))
print("Number of rows for digit '1':", len(indexTo1))
zeroOne = X.loc[np.concatenate((indexTo0, indexTo1))] 
start0 = 0
start1 = len(indexTo0)
```

We compute a truncated SVD of the data at those indices, which are the images with 0's and 1's.

```python
# Compute the truncated SVD of the 0's and 1's of MNIST
n_components = 344 
svd = TruncatedSVD(n_components=n_components)  # Compute the first 'n_components' singular vectors and values
X_svd = svd.fit_transform(zeroOne.T)

U01 = X_svd / svd.singular_values_
S01 = np.diag(svd.singular_values_)
VT01 = svd.components_
```

And build an LDA that classifies the two digits by training on the first 60 images of each and testing on 20 images of each. It uses 2 arbitrary feature columns, since using more would be more computationally costly.

```python
plt.rcParams['figure.figsize'] = [12, 12]
featureCols=[1,3]
xtrain2 = np.concatenate((VT01.T[:60,featureCols],VT01.T[len(indexTo1):(len(indexTo1)+60),featureCols]))
label2 = np.repeat(np.array([1,-1]),60)  # label 1 for "0" and -1 for "1"
test2 = np.concatenate((VT01.T[60:80,featureCols],VT01.T[(len(indexTo1)+60):(len(indexTo1)+80),featureCols]))

lda = LinearDiscriminantAnalysis()
test_class = lda.fit(xtrain2, label2).predict(test2)
acc = (sum(1==test_class[range(20)])+sum(-1==test_class[range(20,40)]))/40
```

We can make a bar graph and a scatter plot of the results.

```python
fig,axs = plt.subplots(2)
axs[0].bar(range(40),test_class)
axs[0].set_title('Bar Graph of LDA Classification for 2 Digits')

axs[1].plot(VT01.T[:start1,1],VT01.T[:start1,3],'ro',markerfacecolor=(0,1,0.2),markeredgecolor='k',ms=8,alpha=.5)
axs[1].plot(VT01.T[start1:,1],VT01.T[start1:,3],'bo',markerfacecolor=(0.9,0,1),markeredgecolor='k',ms=8,alpha=.2)
axs[1].set_xlabel('V1')
axs[1].set_ylabel('V3')
axs[1].set_title('Scatter Plot of LDA Classification for 2 Digits')
plt.show()
```

We similarly do the same with 3 digits, 0, 1, and 2.

We can further see the usefulness of our model by using cross-validation.

```python
from sklearn import svm
from sklearn.model_selection import cross_val_score
trials = 100
Clda = cross_val_score(lda, xtrain3, label3, cv=trials)

plt.figure()
ax = plt.axes()
plt.bar(range(trials),Clda*100)
plt.plot(range(trials),100*np.mean(Clda)*np.ones(trials),'r:',linewidth=3)
ax.set_ylabel('Accuracy(%)')
ax.set_xlabel('Trial')
title = "Mean accuracy of LDA: "+'%.1f' % (100*np.mean(Clda)) + "%"
ax.set_title(title)
print(title)
```

We can also build models using SVM and decision trees using their respective sklearn modules, `svm` and `tree`.

```python
Mdl = svm.SVC(kernel='rbf',gamma='auto').fit(xtrain3,label3)
test_labels = Mdl.predict(test3)
trials = 100
CMdl = cross_val_score(Mdl, xtrain3, label3, cv=trials) #cross-validate the model
classLoss = 1-np.mean(CMdl) # average error over all cross-validation iteration
print("classLoss:",classLoss)
plt.figure()
ax = plt.axes()
plt.bar(range(trials),CMdl*100)
plt.plot(range(trials),100*np.mean(CMdl)*np.ones(trials),'r:',linewidth=3)
ax.set_ylabel('Accuracy(%)')
ax.set_xlabel('Trial')
title = "Mean accuracy for SVM: "+ '%.1f' % (100*np.mean(CMdl)) + "%"
ax.set_title(title)
print(title)
```

```python
from sklearn import tree, preprocessing

decision_tree = tree.DecisionTreeClassifier(max_depth=7).fit(xtrain3,label3)
test_labels = decision_tree.predict(test3)
trials = 100
CDT = cross_val_score(decision_tree, xtrain3, label3, cv=trials) # cross-validate the model

plt.figure()
ax = plt.axes()
plt.bar(range(trials),CDT*100)
plt.plot(range(trials),100*np.mean(CDT)*np.ones(trials),'r:',linewidth=3)
ax.set_ylabel('Accuracy(%)')
ax.set_xlabel('Trial')
title="Mean accuracy for decision tree model: " + '%.1f' % (100*np.mean(CDT)) +"%"
ax.set_title(title)
```

### iii. Classify All Digit Pairs

We use itertools to iterate through every pair of digits and use our LDA model on each pair, comparing the accuracy of each.

```python
# Iterate for all combinations of pairs
j=0
import itertools
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for subset in itertools.combinations(digits, 2):
    i1Rows = np.where(y == subset[0])[0]
    i2Rows = np.where(y == subset[1])[0]
    print("Number of rows for digit ", subset[0], ":", len(i1Rows))
    print("Number of rows for digit ", subset[1], ":", len(i2Rows))

    X2 = X.loc[np.concatenate((i1Rows, i2Rows))]

    i1Start = 0
    i2Start = len(i1Rows)

    # Compute the truncated SVD of MNIST
    n_components = 344 
    svd = TruncatedSVD(n_components=n_components)  # Compute the first 'n_components' singular vectors and values
    X2_svd = svd.fit_transform(X2.T)
    U2 = X2_svd / svd.singular_values_
    S2 = np.diag(svd.singular_values_)
    VT2 = svd.components_

    featureCols=[1,3]
    xtrain2 = np.concatenate((VT2.T[:60,featureCols],VT2.T[len(i1Rows):(len(i1Rows)+60),featureCols]))
    label2 = np.repeat(np.array([1,-1]),60)  # labeled 1 for "0" and -1 for "1"
    test2 = np.concatenate((VT2.T[60:80,featureCols],VT2.T[(len(i1Rows)+60):(len(i1Rows)+80),featureCols]))

    lda = LinearDiscriminantAnalysis()
    test_class = lda.fit(xtrain2, label2).predict(test2)  
    acc = (sum(1==test_class[range(20)])+sum(-1==test_class[range(20,40)]))/40
    print('accuracy:', acc)

    if((j % 16)== 0) : plt.figure(figsize=(12,8))      
    plt.subplot(4,4, (j % 16)+1)
    plt.plot(VT2.T[:i2Start,1],VT2.T[:i2Start,3],'ro',markerfacecolor=(0,1,0.2),markeredgecolor='k',ms=8,alpha=.5)
    plt.plot(VT2.T[i2Start:,1],VT2.T[i2Start:,3],'bo',markerfacecolor=(0.9,0,1),markeredgecolor='k',ms=8,alpha=.2)
    plt.xlabel('V1')
    plt.ylabel('V3')
    title="Accuracy of ("+ subset[0] + ', ' + subset[1] + ') : %.1f' % (100*acc) +"%"
    plt.title(title)
    j = j+1

    plt.suptitle('Pairs of two digits showing ease of separation based on accuracy of LDA on the test data')
    plt.tight_layout()
```

### iv. Classifying All Digits

Similarly, we can classify all the digits together by ordering them by label.
```python
# Re-order rows to prepare for splitting training data from testing data
X09 = X.loc[np.concatenate((np.where(y == '0')[0], np.where(y == '1')[0],np.where(y == '2')[0],np.where(y == '3')[0],np.where(y == '4')[0]
                             ,np.where(y == '5')[0],np.where(y == '6')[0],np.where(y == '7')[0],np.where(y == '8')[0],np.where(y == '9')[0]))] 
```

We take the truncated SVD, and create the training and testing data.
```python
n_components = 344
svd = TruncatedSVD(n_components=n_components)  # Compute the first 'n_components' singular vectors and values
X_svd = svd.fit_transform(X09.T)

U09 = X_svd / svd.singular_values_
S09 = np.diag(svd.singular_values_)
VT09 = svd.components_

featureCols=list()
for i in range(10):
    featureCols.append(np.where(y == str(i))[0][0])
print(featureCols)
start0 = 0
start1 = len(np.where(y == '0')[0])
start2 = start1+len(np.where(y == '1')[0])
start3 = start2+len(np.where(y == '2')[0])
start4 = start3+len(np.where(y == '3')[0])
start5 = start4+len(np.where(y == '4')[0])
start6 = start5+len(np.where(y == '5')[0])
start7 = start6+len(np.where(y == '6')[0])
start8 = start7+len(np.where(y == '7')[0])
start9 = start8+len(np.where(y == '8')[0])
trainRows=1000
xtrain10 = np.concatenate((VT09.T[:trainRows,np.array(featureCols)] 
                         ,VT09.T[start1:(start1+trainRows),np.array(featureCols)]
                         ,VT09.T[start2:(start2+trainRows),np.array(featureCols)]
                         ,VT09.T[start3:(start3+trainRows),np.array(featureCols)]
                         ,VT09.T[start4:(start4+trainRows),np.array(featureCols)]
                         ,VT09.T[start5:(start5+trainRows),np.array(featureCols)]
                         ,VT09.T[start6:(start6+trainRows),np.array(featureCols)]
                         ,VT09.T[start7:(start7+trainRows),np.array(featureCols)]
                         ,VT09.T[start8:(start8+trainRows),np.array(featureCols)]
                         ,VT09.T[start9:(start9+trainRows),np.array(featureCols)]))
label10 = np.repeat(np.array(list(range(10))),trainRows)
print('label10:',label10)
print("xtrain10.shape:", xtrain10.shape)
```

We then use cross-validation for each model using these training and testing images.
```python
## LDA for classifying all 10 different digits
trials = 100
lda = LinearDiscriminantAnalysis()
CLda = cross_val_score(lda, xtrain10, label10, cv=trials) #cross-validate the model
classLoss = 1-np.mean(CMdl) # average error over all cross-validation iteration
print("classLoss:",classLoss)
plt.figure()
ax = plt.axes()
plt.bar(range(trials),CLda*100)
plt.plot(range(trials),100*np.mean(CLda)*np.ones(trials),'r:',linewidth=3)
ax.set_ylabel('Accuracy(%)')
ax.set_xlabel('Trail')
title="Mean accuracy for LDA: " + '%.1f' % (100*np.mean(CLda)) +"%"
ax.set_title(title)

## SVM for classifying all 10 different digits
## WARNING: This can take a very long time for large xtrain10 and many trials...
Mdl = svm.SVC(kernel='rbf',gamma='auto').fit(xtrain10,label10)
trials = 10 
CMdl = cross_val_score(Mdl, xtrain10, label10, cv=trials) #cross-validate the model
classLoss = 1-np.mean(CMdl) # average error over all cross-validation iteration
print("classLoss:",classLoss)
plt.figure()
ax = plt.axes()
plt.bar(range(trials),CMdl*100)
plt.plot(range(trials),100*np.mean(CMdl)*np.ones(trials),'r:',linewidth=3)
ax.set_ylabel('Accuracy(%)')
ax.set_xlabel('Trial')
title="Mean accuracy for SVM: " + '%.1f' % (100*np.mean(CMdl)) +"%"
ax.set_title(title)

## Decision tree for classifying all 10 different digits
decision_tree = tree.DecisionTreeClassifier(max_depth=10).fit(xtrain10,label10)
trials = 100
CDT = cross_val_score(decision_tree, xtrain10, label10, cv=trials) #cross-validate the model

plt.figure()
ax = plt.axes()
plt.bar(range(trials),CDT*100)
plt.plot(range(trials),100*np.mean(CDT)*np.ones(trials),'r:',linewidth=3)
ax.set_ylabel('Accuracy(%)')
ax.set_xlabel('Trial')
title="Mean accuracy for decision tree: " + '%.1f' % (100*np.mean(CDT)) +"%"
ax.set_title(title)
```

From section iii, we found the hardest and easiest pairs to classify. We can use each model on these pairs to compare the accuracy of each.

```python
# Hardest pair: (5, 8)
i1=5
i1Rows = np.where(y == str(i1))[0]
i2=8
i2Rows = np.where(y == str(i2))[0]
X2 = X.loc[np.concatenate((i1Rows, i2Rows))] 
i1Start = 0
i2Start = len(i1Rows)
print("---------------\nHardest pair (",i1,",", i2,")")
print("X2.shape:", X2.shape)
# Compute the truncated SVD of MNIST
n_components = 344 # 
svd = TruncatedSVD(n_components=n_components)  # Compute the first 'n_components' singular vectors and values
X2_svd = svd.fit_transform(X2.T)

U2 = X2_svd / svd.singular_values_
S2 = np.diag(svd.singular_values_)
VT2 = svd.components_
print("VT2.shape:", VT2.shape)
featureCols=[1,3]
xtrain2 = np.concatenate((VT2.T[:60,featureCols],VT2.T[len(indexTo1):(len(indexTo1)+60),featureCols]))
label2 = np.repeat(np.array([1,-1]),60)  # labeled 1 for "5" and -1 for "8"
test2 = np.concatenate((VT2.T[60:80,featureCols],VT2.T[(len(indexTo1)+60):(len(indexTo1)+80),featureCols]))
print("xtrain2.shape:", xtrain2.shape)
print("test2.shape:", test2.shape)
# LDA
lda = LinearDiscriminantAnalysis()
test_class = lda.fit(xtrain2, label2).predict(test2)  
accLDA = (sum(1==test_class[range(20)])+sum(-1==test_class[range(20,40)]))/40

# SVM
Mdl = svm.SVC(kernel='rbf',gamma='auto').fit(xtrain2,label2)
test_class = decision_tree.fit(xtrain2, label2).predict(test2)
accSVM = (sum(1==test_class[range(20)])+sum(-1==test_class[range(20,40)]))/40

## DecisionTree
decision_tree = tree.DecisionTreeClassifier(max_depth=5).fit(xtrain2,label2)
test_class = decision_tree.fit(xtrain2, label2).predict(test2)
accDT = (sum(1==test_class[range(20)])+sum(-1==test_class[range(20,40)]))/40
print("---------------\n")
print('accuracy of LDA:', '%.1f' % (100*accLDA) +"%")
print('accuracy of SVM:', '%.1f' % (100*accSVM) +"%") 
print('accuracy of DecisionTree:', '%.1f' % (100*accDT) +"%")
```

```python
# Easiest pair: (6, 9)
i1=6
i1Rows = np.where(y == str(i1))[0]
i2=9
i2Rows = np.where(y == str(i2))[0]
X2 = X.loc[np.concatenate((i1Rows, i2Rows))] 
i1Start = 0
i2Start = len(i1Rows)
print("---------------\nEasiest pair (",i1,",", i2,")")
print("X2.shape:", X2.shape)
# Compute the truncated SVD of digits 6 and 9 in MNIST
n_components = 344 
svd = TruncatedSVD(n_components=n_components)  # Compute the first 'n_components' singular vectors and values
X2_svd = svd.fit_transform(X2.T)

U2 = X2_svd / svd.singular_values_
S2 = np.diag(svd.singular_values_)
VT2 = svd.components_
print("VT2.shape:", VT2.shape)
featureCols=[1,3]
xtrain2 = np.concatenate((VT2.T[:60,featureCols],VT2.T[len(indexTo1):(len(indexTo1)+60),featureCols]))
label2 = np.repeat(np.array([1,-1]),60)  # labeled 1 for "6" and -1 for "9"
test2 = np.concatenate((VT2.T[60:80,featureCols],VT2.T[(len(indexTo1)+60):(len(indexTo1)+80),featureCols]))
print("xtrain2.shape:", xtrain2.shape)
print("test2.shape:", test2.shape)
# LDA
lda = LinearDiscriminantAnalysis()
test_class = lda.fit(xtrain2, label2).predict(test2)  
accLDA = (sum(1==test_class[range(20)])+sum(-1==test_class[range(20,40)]))/40

# SVM
Mdl = svm.SVC(kernel='rbf',gamma='auto').fit(xtrain2,label2)
test_class = decision_tree.fit(xtrain2, label2).predict(test2)
accSVM = (sum(1==test_class[range(20)])+sum(-1==test_class[range(20,40)]))/40

## DecisionTree
decision_tree = tree.DecisionTreeClassifier(max_depth=5).fit(xtrain2,label2)
test_class = decision_tree.fit(xtrain2, label2).predict(test2)
accDT = (sum(1==test_class[range(20)])+sum(-1==test_class[range(20,40)]))/40
print("---------------\n")
print('accuracy of LDA:', '%.1f' % (100*accLDA) +"%")
print('accuracy of SVM:', '%.1f' % (100*accSVM) +"%") 
print('accuracy of DecisionTree:', '%.1f' % (100*accDT) +"%")
```

## 4. Computational Results

### i. Truncated SVD of MNIST

![image](https://user-images.githubusercontent.com/108769794/234184355-7626fc19-db2d-44de-936c-3124b8cd0f57.png)

![image](https://user-images.githubusercontent.com/108769794/234183520-63d7b5f5-8305-4d31-9133-88d1e3c81704.png)

![image](https://user-images.githubusercontent.com/108769794/234183718-788d028a-7149-4e6c-8267-5726194a6841.png)

![image](https://user-images.githubusercontent.com/108769794/234183737-72c77397-cbe1-4b37-a61d-0e911f5b863a.png)

**Rank of MNIST: 713**

![image](https://user-images.githubusercontent.com/108769794/234183759-aa7d0cbb-268e-470a-af72-0d6109083349.png)

![image](https://user-images.githubusercontent.com/108769794/234183775-5b3d7190-2b51-4e3c-b38f-936d817d45fd.png)

![image](https://user-images.githubusercontent.com/108769794/234183796-d98d5317-a339-4c46-9ca6-d34dcd575581.png)

### ii. Building A Linear Classifier

![image](https://user-images.githubusercontent.com/108769794/234183824-cc7a90dc-900f-4b7b-b233-d349b0ed4be7.png)

![image](https://user-images.githubusercontent.com/108769794/234183868-7876a8b3-4729-4b12-b3f2-91ffdd74cd8b.png)

![image](https://user-images.githubusercontent.com/108769794/234183893-e1f69f0e-1c9a-47ff-8e65-68f35cc3d9fd.png)

![image](https://user-images.githubusercontent.com/108769794/234183916-2c5b8608-8057-464f-beb7-aee70ac817fd.png)

![image](https://user-images.githubusercontent.com/108769794/234183927-1e89e04a-0343-43bb-b23d-8084d593f97a.png)

### iii. Classifying All Digit Pairs

![image](https://user-images.githubusercontent.com/108769794/234183975-c9dbfaa4-80dd-4dea-80ae-bdf58597f873.png)

![image](https://user-images.githubusercontent.com/108769794/234184000-d0497e80-0a75-467e-9d25-066efeaca454.png)

![image](https://user-images.githubusercontent.com/108769794/234184009-a1e1f72f-d6e5-43c2-a69c-166ca85040c3.png)

### iv. Classifying All Digits

![image](https://user-images.githubusercontent.com/108769794/234184058-b8b727eb-4d19-452c-9adc-0dd91e7664de.png)

![image](https://user-images.githubusercontent.com/108769794/234184090-08c8518d-e963-4e25-8830-6d27fcb5330f.png)

![image](https://user-images.githubusercontent.com/108769794/234184105-9da6b839-63a5-44be-9e53-d17102f06bde.png)

### v. Comparing Classifier Models

Hardest Pair (5, 8):

accuracy of LDA: 40.0%

accuracy of SVM: 72.5%

accuracy of DecisionTree: 67.5%


Easiest Pair (6, 9):

accuracy of LDA: 100.0%

accuracy of SVM: 100.0%

accuracy of DecisionTree: 100.0%

## 5. Summary and Conclusions

### i. Truncated SVD of MNIST

To capture at least 90% of the data, we need 344 modes. The other 10% is not as important to the data, so we won't lose too much information if we ignore it. Comparing the truncated matrix with the original matrix, we see that the truncated matrix is blurrier, but we can still distinguish each digit. The loss matrix images are mostly noise, meaning the 10% we lost did not account for much.

### ii. Building A Classifier

For classifying 2 digits with LDA, training on 60 points and testing on 20, we get an accuracy of 100%. For classifying 3 digits, there are some incorrect labels, but it is mostly accurate, and we can see from the scatter plot that there are 3 distinct clusters. Cross-validating for the 3 digit classification, we get 85.3% mean accuracy.

Using the same 3 digits, training, and testing data, the SVM model recieves an mean accuracy of 77.5%, and runs much slower than the LDA model.

Now using a decision tree, the model is much more accurate, with a mean accuracy of 93.4%. It is only slightly slower than the LDA model.

### iii. Classifying All Digit Pairs

Using LDA to classify all the pairs of digits, we find the most difficult pairs to classify are 5 and 8, with an accuracy of 52.5%. Visually, they have many similarities, so it makes sense for the model to have difficulty distinguishing them. The easiest pairs are (2, 4), (2, 9), (3, 6), (6, 8), and (6, 9), all with an accuracy of 100%. The more visually distinct each digit is, the easier it is to classify them. This can be observed from the scatter plots, with the distance between clusters of more distinct digit pairs being greater than the similar digit pairs. For example, the clusters of (5, 8) have a large amount of overlap, while (2, 4) are more separated.

### iv. Classifying All Digits

Cross-validating using each model on all the digits, the LDA model has a mean accuracy of 75.2% and the decision tree had a mean accuracy of 73.1%. The SVM model took a long time to run, even on only 10 trials, and only reached an mean accuracy of 62.9%.

### v. Comparing Classifier Models

On classifying the hardest pair, (5, 8), the decision tree did the best, while LDA did the worst. On classifying one of the easiest pairs (6, 9), all models have 100% accuracy.

Overall, the decision tree model was the most accurate, while SVM struggled to run on such a large dataset.
