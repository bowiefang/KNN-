#!/usr/bin/env python
# coding: utf-8

# In[3]:


from csv import reader
from sys import exit
from math import sqrt
from operator import itemgetter
import numpy as np
import pandas as pd
from IPython.display import Image
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# In[4]:


iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
iris.tail()


# In[5]:


# 给数据加上列名
iris = iris.sample(frac=1).reset_index(drop=True)
iris.columns = (['Sepal Length', 'Sepal Width', 'Petal Length' , 'Petal Width','type'])
iris.head(6)


# In[6]:


iris.describe()


# In[7]:


X = np.array(iris)[:, :4]  # we only take the first two features.
X = X.astype(float)
Y = np.array(iris)[:,-1]


# In[8]:


plt.figure(figsize=(8, 6))

# Plot the training points
for (x,y) in zip(X,Y):
    if y=='Iris-setosa':
        c='r'
        plt.scatter(x[0], x[1],color=c, marker='.')
    if y=='Iris-versicolor':
        c='g'
        plt.scatter(x[0], x[1],color=c, marker='x')
    if y=='Iris-virginica':
        c='b'
        plt.scatter(x[0], x[1],color=c, marker='*')        
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()


# In[9]:


def performance (real, predict):
    label_predict = np.array(predict)
    label_real = np.array(real)
    error=0
    for t, p in zip(label_real, label_predict): 
        if t!= p:
            error += 1
    return error/len(label_real)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(
    iris.values[:,:-1], iris.values[:,-1], test_size=0.33, random_state = 1)


knn = KNeighborsClassifier(n_neighbors = 10, algorithm='kd_tree', weights='distance')
knn.fit(X_train, y_train)
print (performance(y_test, knn.predict(X_test)))


# In[11]:


## 读入数据和数据预处理


# In[12]:


from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 数据预处理
train_set, test_set = train_test_split(iris, test_size=0.3, random_state=1)


# In[13]:


## 模型构建


# In[14]:


def get_classes(y):
    return np.unique(y)

def find_neighbors(distances, k):
    return distances[0:k]

def find_response(neighbors, classes):
    votes = {c:0 for c in classes}
    for instance in neighbors:
        for c in classes:
            if instance[0] == c:
                votes[c] += 1
    return max(votes,key=votes.get)

def knn(x_train, y_train, x_test, k):
    dist = 0
    classes = get_classes(y_train)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    # generate response classes from training data
    predict=[]
    for test_instance in x_test:
        distances = []
        for (feature, label) in zip(x_train,y_train):
            test_instance = test_instance.astype('float')
            feature = feature.astype('float')
            dist = np.sum((test_instance - feature)**2)
            distances.append([str(label),sqrt(dist)])
            dist = 0
        # 按照最后一列排序    
        distances.sort(key = lambda x: x[-1])
        # find k nearest neighbors
        neighbors = find_neighbors(distances, k)
        value = find_response(neighbors, classes)
        predict.append(value)  
    return predict  


# In[15]:


K的选择
为了选择K，把数据分割成训练集和测试集
k-fold 交叉验证
k-fold　交叉验证中选择　n_splits = 5, 选择合适大小的k, knn 算法中的选取　k nearest　neighbours, 是的返回该分类方法的平均误差最少。


# In[16]:


def test(k,data):
    kf = KFold(n_splits = 5)
    kf.get_n_splits(train_set)
    Error = 0
    for train_index, test_index in kf.split(data.values[:,-1]):
        X,ｙ = data.values[:,:-1], data.values[:,-1]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        predict=knn(X_train, y_train, X_test, k)
        error = performance(y_test, predict)
        Error += error
    return Error/5   


# In[17]:


plt.figure(0)
for k in range(10):
    plt.scatter(k + 1, test(k + 1, iris),color = 'red')
plt.xlabel('k')
plt.ylabel('error rate')
plt.show()    


# In[18]:


types = np.unique( iris.values[:,-1])
iris_noisy = iris.copy()
for i in range(10):
    index = np.random.randint(0,3)
    iris_noisy.iloc[np.random.randint(0,len(iris)),-1] = types[index]


# In[19]:


X = np.array(iris_noisy)[:, :4]  # we only take the first two features.
X = X.astype(float)
Y = np.array(iris_noisy)[:,-1]

plt.figure(figsize=(8, 6))

# Plot the training points
for (x,y) in zip(X,Y):
    if y=='Iris-setosa':
        c='r'
        plt.scatter(x[0], x[1],color=c, marker='.')
    if y=='Iris-versicolor':
        c='g'
        plt.scatter(x[0], x[1],color=c, marker='x')
    if y=='Iris-virginica':
        c='b'
        plt.scatter(x[0], x[1],color=c, marker='*')        
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()


# In[20]:


plt.figure(0)
for k in range(10):
    plt.scatter(k + 1, test(k + 1, iris_noisy),color = 'red')
plt.xlabel('k')
plt.ylabel('error rate')
plt.show() 


# In[ ]:




