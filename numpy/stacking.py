import numpy as np

rg = np.random.default_rng(1)
a = np.floor(10 * rg.random((2, 2)))

print("a", a)

b = np.floor(10 * rg.random((2, 2)))
print("b", b)

# vstack 表示垂直拼接
print("vstack", np.vstack((a, b)))
# hstack 表示水平拼接
print("hstack", np.hstack((a, b)))

# column_stack 将一维或二维数组按列进行拼接
print("column_stack",np.column_stack((a, b)))

# 一组数组（必须具有相同的元素数量或行数），一个按列堆叠后的新数组。
# np.column_stack 会将一维数组转换成列向量（形状变为 (n,1)），再按列拼接。
# 列向量表示数组每个元素作为一列单独存在。
test1 = np.array([1, 2, 3])
test2 = np.array([4, 5, 6])
print("test1", np.column_stack((test1, test2)))
# [[1 4]
#  [2 5]
#  [3 6]]

# 二维数组（会直接按列进行拼接）,与 hstack 效果相同

# 数组拼接快捷方式
print(np.r_[1:4, 0, 4]) # [1 2 3 0 4]
# 1:4
#
# 等价于 np.arange(1, 4)。
# 生成 [1, 2, 3]。
# 0
#
# 单个值 0，作为一个元素添加到数组中。
# 4
#
# 单个值 4，作为一个元素添加到数组中。

# np.r_[1:4, 0, 4] 的等价写法： result = np.concatenate([np.arange(1, 4), [0], [4]])



