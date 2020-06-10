#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @ProjectName :Recommendation
# @FileName    :testRead.py
# @Time        :2020/6/10  18:27
# @Author      :Shuhao Chen
import codecs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
def loadMovieLensTrain( ):
	f = codecs.open('D:/数据挖掘Project/train.txt', encoding='UTF-8')
	data = pd.read_csv(f, header=None, sep="\s+", names=['user', 'item', 'score'])

	prefer = {}
	for line in data.itertuples():  # 打开指定文件
		(userid, movieid, rating) = (line[1] , line[2] ,line[3])
		prefer.setdefault(userid, {})  # 设置字典的默认格式,元素是user:{}字典
		prefer[userid][movieid] = float(rating)

	return prefer  # 格式如{'user1':{itemid:rating, itemid2:rating, ,,}, {,,,}}


##==================================
#        加载对应的测试集文件
#  参数fileName 代表某个测试集文件,如u1.test
##==================================
def loadMovieLensTest():
	f = codecs.open('D:/数据挖掘Project/train.txt', encoding='UTF-8')
	data = pd.read_csv(f, header=None, sep="\s+", names=['user', 'item', 'score'])

	prefer = {}
	m = data.user.nunique()
	n = 1682  # 因为数据有缺失
	userItemMatrix = np.zeros((m, n))
	for line in data.itertuples():
		userItemMatrix[line[1] - 1, line[2] - 1] = line[3]  # 矩阵从0开始索引，user和item从1开始索引
	for i in range(len(userItemMatrix)):
		for j in range(len(userItemMatrix[i])):
			(userid, movieid, rating) = (i+1, j+1, userItemMatrix[i][j])
			prefer.setdefault(userid, {})  # 设置字典的默认格式,元素是user:{}字典
			prefer[userid][movieid] = float(rating)

	return prefer


if __name__ == "__main__":
	print("""这个部分可以进行上面2个函数测试 """)

	trainDict = loadMovieLensTrain()
	testDict = loadMovieLensTest()

	print(len(trainDict))
	print(len(testDict))

	print(""" 测试通过 """)
