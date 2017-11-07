import numpy as np

import matplotlib.pyplot as plt
import timeit

# data1 = [6, 7.5, 8, 0, 1]
# arr1 = np.array(data1, dtype=np.float32)
# print(arr1)
# print(arr1.dtype)
#
# data2 = [[1, 2, 3, 4], [5, 6, 7,8]]
# arr2 = np.array(data2, dtype=np.int64)
# print(arr2)
# print(arr2.ndim)
# print(arr2.shape)
# print(arr2.dtype)
#
# print(np.zeros(10))
# print(np.zeros((3, 6)))
# print(np.empty((2, 3, 2)))
# print(np.arange(15))
#
# arr = np.array([1, 2,3, 4])
# print(arr.dtype)
# float_arr = arr.astype(np.float64)
# print(float_arr.dtype)

# arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
# print(arr.astype(np.int32))
#
# numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
# print(numeric_strings.astype(float))

# int_array = np.arange(10)
# calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
# print(int_array.astype(calibers.dtype))

# arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# print(arr)
# print(arr * arr)
# print(arr + arr)
# print(arr - arr)
# print(1/arr)

# arr = np.arange(10)
# print(arr)
# print(arr[5])
# print(arr[5:8])
# arr[5:8] = 12
# print(arr)
#
# arr_slice = arr[5:8]
# arr_slice[1] = 12345
# print(arr)
# arr_slice[:] = 64
# print(arr)

# arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(arr2d[2])
# print(arr2d[0][2])
# print(arr2d[0, 2])

# arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# print(arr3d)
# print(arr3d[0])
# old_values = arr3d[0].copy()
# arr3d[0] = 42
# print(arr3d)
# arr3d[0] = old_values
# print(arr3d)
# print(arr3d[1, 0])

# names = np.array(['bob', 'joe', 'will', 'bob', 'will', 'joe', 'joe'])
# data = np.random.randn(7, 4)
# print(names)
# print(data)
# print(names == 'bob')
# print(data[names == 'bob'])
# print(data[names == 'bob', 2:])
# print(data[names == 'bob', 3])
# print(names != 'bob')
# print(data[names != 'bob'])
# print(data[~(names == 'bob')])
# mask = (names == 'bob') | (names == 'will')
# print(mask)
# print(data[mask])
# data[data < 0] =0
# print(data)
# data[names != 'joe'] = 7
# print(data)

# arr = np.empty((8, 4))
# for i in range(8):
#     arr[i] = i
# print(arr)
# print(arr[[4, 3, 0, 6]])
# print(arr[[-3, -5, -7]])

# arr = np.arange(32).reshape((8, 4))
# print(arr)
# print(arr[[1, 5, 7, 2], [0, 3, 1, 2]])
# print(arr[[1, 5, 7, 2]][:, [0, 3, 1,2]])
# print(arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])

# arr = np.arange(15).reshape((3, 5))
# print(arr)
# print(arr.T)

# arr = np.random.randn(6, 3)
# print(np.dot(arr.T, arr))

# arr = np.arange(16).reshape((2, 2, 4))
# print(arr)
# print(arr.transpose((1, 0, 2)))
# print(arr.swapaxes(1, 2))

# arr = np.arange(10)
# print(np.sqrt(arr))
# print(np.exp(arr))

# x = np.random.randn(8)
# y = np.random.randn(8)
# print(x)
# print(y)
# # 元素级最大值
# print(np.maximum(x, y))

# arr = np.random.randn(7) * 5
# print(np.modf(arr))

# 1000个间隔相等的点
# points = np.arange(-5, 5, 0.01)
# xs, ys = np.meshgrid(points, points)
# # print(ys)
# z = np.sqrt(xs ** 2 + ys ** 2)
# print(z)
# plt.imshow(z, cmap=plt.cm.gray)
# plt.title("Image plot of $\sqrt{x^2+y^2}$ for a grid of values")
# plt.savefig("data analysize")

# xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
# yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
# cond = np.array([True, False, True, True, False])
# result = [(x if c else y)
#             for x, y, c in zip(xarr, yarr, cond)]
# print(result)
# result = np.where(cond, xarr, yarr)
# print(result)

# arr = np.random.randn(4, 4)
# print(arr)
# # result = np.where(arr > 0, 2, -2)
# # print(result)
# result = np.where(arr > 0, 2, arr)
# print(result)
