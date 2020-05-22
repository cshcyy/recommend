#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @ProjectName :Recommendation
# @FileName    :test.py
# @Time        :2020/5/22  15:18
# @Author      :Shuhao Chen
import codecs
import pandas as pd
import numpy as np

n = np.arange(0, 30, 2)# start at 0 count up by 2, stop before 30
n = n.reshape(3, 5) # reshape array to be 3x5
d= np.mat(n.nonzero())
x = n[n.nonzero()]  # 取出矩阵中的非零元素的坐标

print(d)
print(x)
f = codecs.open('D:/数据挖掘Project/train.txt', encoding='UTF-8')
data = pd.read_csv(f, header=None, sep="\s+", names=['user', 'item', 'score'])
a=data.itertuples()
print(type(data))
print(type(a))