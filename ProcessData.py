#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @ProjectName :Recommendation
# @FileName    :ProcessData.py
# @Time        :2020/5/22  14:57
# @Author      :Shuhao Chen
import codecs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

f = codecs.open('D:/数据挖掘Project/train.txt', encoding='UTF-8')
data = pd.read_csv(f, header=None, sep="\s+", names=['user', 'item', 'score'])
# 将user数当作行数，item数当作列数创建一个m*n的矩阵
m = data.user.nunique()
n = 1682  # 因为数据有缺失
userItemMatrix = np.zeros((m, n))

for line in data.itertuples():
	userItemMatrix[line[1] - 1, line[2] - 1] = line[3]  # 矩阵从0开始索引，user和item从1开始索引

# 计算非零元素的个数与矩阵大小相除得到稀疏性#计算完后发现为0.038，非常稀疏。
sparsity = round(len(userItemMatrix.nonzero()[1]) / float(m * n), 3)
# print(sparsity)

# 接下来开始构建训练集和测试集
trainData, testData = train_test_split(data, test_size=0.3)  #测试样本占比0.3
trainDataMatrix = np.zeros((m, n))
testDataMatrix = np.zeros((m, n))
for line in trainData.itertuples():
	trainDataMatrix[line[1] - 1, line[2] - 1] = line[3]       #将data变为元组形式一个一个存进矩阵中

for line in testData.itertuples():
	testDataMatrix[line[1] - 1, line[2] - 1] = line[3]

# 基于SVD,即奇异值分解方法实现协同过滤推荐系统
u,s,vt = svds(trainDataMatrix,k=20)
s_diag_matrix = np.diag(s) # 将s转化为对角阵
svd_prediction = np.dot(np.dot(u,s_diag_matrix),vt)
print(s)
print(u.shape)
print(s.shape)
print(vt.shape)
print(s_diag_matrix.shape)
print(svd_prediction.shape)
svd_prediction[svd_prediction < 1] = 1
svd_prediction[svd_prediction > 5] = 5

#训练集预测

from sklearn.metrics import mean_squared_error
from math import sqrt
predictionFlatten = svd_prediction[trainDataMatrix.nonzero()] #取出所有非零值
trainDataMatrixFlatten = trainDataMatrix[trainDataMatrix.nonzero()]
trainError = sqrt(mean_squared_error(predictionFlatten,trainDataMatrixFlatten))
print('训练集预测均方根误差：', '\n' , trainError)
print(predictionFlatten.shape)
#测试集预测
predictionFlatten = svd_prediction[testDataMatrix.nonzero()]
testDataMatrixFlatten = testDataMatrix[testDataMatrix.nonzero()]
testError = sqrt(mean_squared_error(predictionFlatten,testDataMatrixFlatten))
print('测试集预测均方根误差', testError)
print(predictionFlatten.shape)