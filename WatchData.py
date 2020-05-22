#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# @ProjectName :Recommendation
# @FileName    :WatchData.py
# @Time        :2020/5/22  14:55
# @Author      :Shuhao Chen
import codecs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

f = codecs.open('D:/数据挖掘Project/train.txt', encoding='UTF-8')
data = pd.read_csv(f, header=None, sep="\s+", names=['user', 'item', 'score'])
print(data.head())  # 查看原始数据
print(data.info())  # 查看数据集信息
print(data.describe())  # 查看数据描述，均值方差等
print(data.user.nunique())  # 查看uesr和item的个数，发现user943个，item1617个
print(data.item.nunique())
print(data.duplicated(subset=['user', 'item']).sum())  # 检测useritem会不会出现重复，结果为0，说明没有重复，数据清洗过了

itemToUser = data.groupby('item').count().user  # 统计每个物品对应的用户数
plt.hist(itemToUser.values)  # 做直方图展示
plt.show()
quanItem = itemToUser.quantile(q=np.arange(0, 1.1, 0.1)) #查看每个item 对应的 user的十分位数到百分位数
print(quanItem)  #发现到了百分30的物品对应的用户就只有6个了
userToItem = data.groupby('user').count().item
plt.hist(userToItem.values)
plt.show()
quanUser = userToItem.quantile(q=np.arange(0,1.1,0.1))
print(quanUser)