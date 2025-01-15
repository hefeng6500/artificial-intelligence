import numpy as np

rg = np.random.default_rng(1)
a = np.floor(10 * rg.random((2, 12)))

print(a)

# hsplit 水平分割，不修改原数组
print(np.hsplit(a, 3))

# vsplit 垂直分割，不修改原数组
print(np.vsplit(a, 2))

# split 分割，不修改原数组
print(np.split(a, 3, axis=1))
