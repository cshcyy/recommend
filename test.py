#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @ProjectName :Recommendation
# @FileName    :test.py
# @Time        :2020/5/22  15:18
# @Author      :Shuhao Chen
import codecs
import pandas as pd
import numpy as np

# n = np.arange(0, 30, 2)  # start at 0 count up by 2, stop before 30
# n = n.reshape(3, 5)  # reshape array to be 3x5
# d = np.mat(n.nonzero())
# x = n[n.nonzero()]  # 取出矩阵中的非零元素的坐标
# n[n>3] = 5
# print(n)
#
# f = codecs.open('D:/数据挖掘Project/train.txt', encoding='UTF-8')
# data = pd.read_csv(f, header=None, sep="\s+", names=['user', 'item', 'score'])
# a = data.itertuples()
# print(type(data))
list = [2] * 4
list[1] = 5
length = len(list)
for i in range(length):
	print(list[i])
list.sort()
print(list)
for i in range(len(list) - 2, -1, -1):
	print(i)

# a,b=b,a+b的情形
a = 0
b = 1
a, b = b, a + b
print(a, b)
my_dict = {}
my_dict['sex'] = 'male'
my_dict['age'] = 20
print(my_dict)