import numpy as np

a = np.arange(12) ** 2
i = np.array([1, 1, 3, 8, 5])

print(a)
print(a[i])  # the elements of `a` at the positions `i`

j = np.array([[3, 4], [9, 7]])
print(a[j])
# a[j] 将索引数组 j 中的每个元素视为在 a 中的索引。结果数组的形状与索引数组 j 一致：
# [[ 9 16]
#  [81 49]]


# 调色板
palette = np.array([[0, 0, 0],  # black
                    [255, 0, 0],  # red
                    [0, 255, 0],  # green
                    [0, 0, 255],  # blue
                    [255, 255, 255]])  # white
# 图片
image = np.array([[0, 1, 2, 0],  # each value corresponds to a color in the palette
                  [0, 3, 4, 0]])

print(palette[image])
# [[[  0   0   0]
#   [255   0   0]
#   [  0 255   0]
#   [  0   0   0]]
#
#  [[  0   0   0]
#   [  0   0 255]
#   [255 255 255]
#   [  0   0   0]]]

