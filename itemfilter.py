#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @ProjectName :Recommendation
# @FileName    :itemfilter.py
# @Time        :2020/5/25  20:50
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
np.savetxt('userItemMatrix.csv', userItemMatrix, delimiter=',',fmt='%0.3f')
# print(sparsity)

# 接下来开始构建训练集和测试集
trainData, testData = train_test_split(data, test_size=0.3)  #测试样本占比0.3
trainDataMatrix = np.zeros((m, n))
testDataMatrix = np.zeros((m, n))
for line in trainData.itertuples():
	trainDataMatrix[line[1] - 1, line[2] - 1] = line[3]       #将data变为元组形式一个一个存进矩阵中

for line in testData.itertuples():
	testDataMatrix[line[1] - 1, line[2] - 1] = line[3]

# 构建物品相似性矩阵，采用余弦距离计算
from sklearn.metrics.pairwise import pairwise_distances
itemSimilarity = pairwise_distances(userItemMatrix.T,metric='cosine')
# print(itemSimilarity)
print(len(itemSimilarity),len(itemSimilarity[0]))
# 对相似矩阵上三角进行分析
itemSimilarityTriu = np.triu(itemSimilarity,k=1) #取得上三角矩阵
itemSimilarityNonzero = np.round(itemSimilarityTriu[itemSimilarityTriu.nonzero()],3)
a = np.percentile(itemSimilarityNonzero,np.arange(0,101,10))
print(a)
# print(itemSimilarityTriu)
# 训练集预测
userItemPrecdiction = userItemMatrix.dot(itemSimilarity)/np.array([np.abs(itemSimilarity).sum(axis=1)])
print(len(userItemPrecdiction),len(userItemPrecdiction[0]))
print(len(trainDataMatrix),len(trainDataMatrix[0]))
# 除以np.array是为了让评分在1-5之间
from sklearn.metrics import mean_squared_error
from math import sqrt
predictionFlatten = userItemPrecdiction[trainDataMatrix.nonzero()]
userItemMatrixFlatten = userItemMatrix[trainDataMatrix.nonzero()]
trainError = sqrt(mean_squared_error(predictionFlatten,userItemMatrixFlatten))
print('训练集均方根误差：', trainError)
np.savetxt('itemfilter.csv', userItemPrecdiction, delimiter=',',fmt='%0.3f')
# 测试集预测

