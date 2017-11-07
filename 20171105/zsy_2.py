import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr
from random import normalvariate
# arr = np.random.randn(5, 4)
# print(arr.mean())
# print(arr.sum())
# print(arr.mean(axis=1))
# print(arr.sum(0))
# arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# print(arr.cumsum(0))
# print(arr.cumprod(1))

# arr = np.random.randn(100)
# 正值的数量
# print((arr > 0).sum())
# 两个方法any和all
# bools = np.array([False, False, True, True, True])
# print(bools.any())
# print(bools.all())
# 排序
# arr = np.random.randn(8)
# print(arr)
# arr = np.random.randn(5, 3)
# print(arr)
# print("\n排序axis=1:")
# print(np.sort(arr, axis=1))
# large_arr = np.random.randn(1000)
# large_arr1 = large_arr.sort()
# 5%分位数
# print(large_arr[int(0.05*len(large_arr))])
# np.unique用于找出数组中的唯一值并返回已排序的结果
# names = np.array(['bob', 'joe', 'will', 'bob', 'will', 'joe', 'joe'])
# print(np.unique(names))
# ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
# print(np.unique(ints))
# 纯python代码
# print(sorted(set(names)))
# 函数np.in1d测试一个数组中的值在另一个数组中的成员资格，返回一个布尔型数组
# values = np.array([6, 0, 0, 3, 2, 5, 6])
# print(np.in1d(values, [2, 3, 6]))
# 将数组以二进制格式保存到磁盘
# arr = np.arange(10)
# np.save('some_array', arr)
# print(np.load('some_array.npy'))
# np.savez('array_archive.npz', a=arr, b=arr)
# arch = np.load('array_archive.npz')
# print(arch['b'])
# 存取文本文件
# a = np.arange(100).reshape((5, 20))
# np.savetxt('array_ex.txt', a, fmt='%d', delimiter=',')
# arr = np.loadtxt('array_ex.txt', delimiter=',')
# print(arr)
# 线性代数
# x = np.array([[1., 2., 3.], [4., 5., 6.]])
# y = np.array([[6., 23.], [-1, 7], [8, 9]])
# print(x)
# print(y)
# print(x.dot(y)) # 相当于np.dot(x,y)
# print(np.dot(x, np.ones(3)))
# INV求逆
# X = np.random.randn(5, 5)
# mat = X.T.dot(X)
# print(inv(mat))
# print(mat.dot(inv(mat)))
# 求行列式
# q, r = qr(mat)
# print(r)
# 随机数生成
# samples = np.random.normal(size=(4, 4))
# print(samples)
# 随机漫步
import random
# position = 0
# walk = [position]
# steps = 1000
# for i in range(steps):
#     step = 1 if random.randint(0, 1) else -1
#     position += step
#     walk.append(position)
# print(walk[:100])
# x = walk[:100]
# y = walk[:100]
# 绘制图像
# plt.figure(figsize=(8, 4))
# 在当前绘图对象绘图
# plt.plot(x, y, linewidth=1)
# plt.show()
# 模拟一次性随机产生1000个”掷硬币“
# nsteps = 1000
# draws = np.random.randint(0, 2, size=nsteps)
# steps = np.where(draws > 0, 1, -1)
# walk = steps.cumsum()
# print(walk)
# print(walk.min())
# print(walk.max())
# print((np.abs(walk) >= 10).argmax())
# 一次模拟多个随机漫步
nwalks = 5000
nsteps = 1000
# 0或1
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
print(walks)
print(walks.max())
print(walks.min())
hits30 = (np.abs(walks) >= 30).any(1)
print(hits30)
print(hits30.sum())
crossing_times = (np.abs(walks[hits30]) >=30).argmax(1)
print(crossing_times.mean())
