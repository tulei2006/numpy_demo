#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/2/11 17:32
# @Author  : GuoChang
# @Site    : https://github.com/xiphodon
# @File    : numpy_demo.py
# @Software: PyCharm Community Edition


# numpy库学习笔记


import numpy as np

def demo_01():
    '''
    读取txt文件
    :return:
    '''
    # 读取txt文件
    all_data_01 = np.genfromtxt("datingTestSet.txt", delimiter="\t")
    # array类型，内部数据类型必须一致
    print(all_data_01)

    # 制表格分割，数据均为str类型
    all_data_02 = np.genfromtxt("datingTestSet.txt",dtype=str, delimiter="\t")
    print(all_data_02)

    # 制表格分割，数据均为Unicode编码，跳过头第一行数据（一般为类别）
    all_data_03 = np.genfromtxt("datingTestSet.txt",dtype='U75', delimiter="\t", skip_header=1)
    print(all_data_03)

    # 输出第三行，第四列
    print(all_data_03[2,3])

def demo_02():
    '''
    向量和矩阵
    :return:
    '''

    # 向量
    vector = np.array([1,2,3,4,5])
    # 矩阵
    matrix = np.array([
        [1,2,3,4,5],
        [6,7,8,9,10],
        [11,12,13,14,15]
    ])

    print(vector)
    print(matrix)

    # 支持切片（包含前不包含后）
    print(vector[1:3])
    print(matrix[0:2,4])    # 前两行的第5列
    print(matrix[:,3])  # 所有行的第4列
    print(matrix[0:2,2:4])  # 前两行的第三四列
    print(type(matrix[:,4]))

    # 向量/矩阵 的维度
    print(vector.shape)
    print(matrix.shape)

    # array内存值得类型，array内必须相同类型
    print(vector.dtype)
    print(matrix.dtype)

    print(vector == 4)  # 逐个判断向量中元素是否等于4，并返回布尔值得向量
    print(matrix%5 == 0)    #逐步判断矩阵中元素是否为5的倍数，并返回对应位置的布尔值矩阵

    # 元素选择
    print(vector[vector == 4])  # 将上述布尔向量代回向量，选择出对应True的元素组成数组
    print(vector[(vector == 2) | (vector == 4)])  # 将上述布尔向量代回向量，选择出对应True的元素组成数组
    print(matrix[matrix%5 == 0])    # 将上述布尔矩阵代回矩阵，选择出对应True的元素组成数组
    print(matrix[[True,False,False],[False,False,True,True,False]]) # 选择第一行的第三四列元素组成的数组

    # 元素选择后赋值
    vector[(vector == 2) | (vector == 4)] = 17
    print(vector)
    matrix[matrix % 5 == 0] = -1
    print(matrix)


def demo_03():
    '''
    转型与计算
    :return:
    '''

    vector = np.array(['1','2','3'])
    print(vector)
    print(vector.dtype) # 内部元素类型
    # 向量内元素转型
    vector = vector.astype(dtype=float)  # 字符串类型转型为浮点型
    print(vector)
    print(vector.dtype)

    # 向量
    vector = np.array([1, 2, 3, 4, 5])
    # 矩阵
    matrix = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15]
    ])

    print(vector)
    print(matrix)

    # 计算向量，求和，最大值，最小值，均值
    print(vector.sum())
    print(vector.max())
    print(vector.min())
    print(vector.mean())

    # 计算矩阵，指定维度求和
    print(matrix.sum(axis=1))   # axis=1 按行求和
    print(matrix.sum(axis=0))   # axis=0 按列求和

if __name__ == "__main__":
    # demo_01()
    # demo_02()
    demo_03()