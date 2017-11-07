import numpy as np
# [3, 2, 3]3行2列3个平面
data = np.random.randint(0, 5, [3, 2, 3])
print(data)
# 默认对最大的axis进行排序，这里即是axis=2
print("默认对最大的axis排序：")
print(np.sort(data))
print("axis=0:")
print(np.sort(data, axis=0))
print("axis=1:")
print(np.sort(data, axis=1))
print("axis=2:")
print(np.sort(data, axis=2))
print("axis=none:")
print(np.sort(data, axis=None))
