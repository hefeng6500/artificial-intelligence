import numpy as np

a = np.array([1, 2, 3])

print(a)
print(a.dtype) # 类型
print(a.ndim) # 维度
print(a.size) # 元素个数
print(a.itemsize) # 每个元素的字节数
print(a.nbytes) # 总字节数

b = np.arange(30).reshape(3, 5, 2)
print(b)
print(type(b)) # type 表示 ndarray 的类型

# 1.创建
c = np.array([1, 2, 3])
print(c.dtype)
d = np.array([1.2, 3.5, 5.1])
print(d.dtype)
e = np.array(['a', 'b', 'c'])
print(e.dtype)

f = np.array([(1.5, 2, 3), (4, 5, 6)])
print(f)
# [
#   [1.5 2.  3. ]
#   [4.  5.  6. ]
# ]
print(f.dtype)

g = np.array([[1.8, 2], [3, 4]], dtype=complex)
print(g)
# [[1.8+0.j 2. +0.j]
#  [3. +0.j 4. +0.j]]
# +0.j 强调了虚部为 0，而 .j 是虚数单位的标准表示。在数学上，这个复数就等价于纯实数 1 或 2，但是为了与复数的标准表示一致，NumPy 会始终展示虚部，即使它是零

# 2.zero
h = np.zeros((3, 4))
print(h)

# 3.range
i = np.arange(10, 30, 5) # 10 到 30，间隔 5
print(i)

j = np.arange(0, 2, 0.3) # 0 到 2，间隔 0.3
print(j)

# 4.linspace
k = np.linspace(0, 2, 9) # 0 到 2，9 个元素
print(k)
l = np.linspace(0, 2 * np.pi, 100)
m = np.sin(l)
print(l)
print(m)

# 5.print
print(np.arange(10000)) # 省略中间部分 [  0   1   2 ... 9997 9998 9999]

# 6.基本操作
n1 = np.array([20, 30, 40, 50])
n2 = np.arange(4) # [0 1 2 3]
n3 = n1 - n2
print(n3) # [20 29 38 47]
n4 = n2 ** 2
print(n4) # [0 1 4 9]
n5 = 10 * np.sin(n1)
print(n5) # [ 9.12945251 -9.88031624  7.4511316   9.21034037]
n6 = n1 < 35
print(n6) # [ True  True False False]

o1 = np.array([[1, 1], [0, 1]])
o2 = np.array([[2, 0], [3, 4]])
# 矩阵乘法
o3 = o1 * o2
print(o3)
# 矩阵点乘
o4 =o1 @ o2 # 或者 o1.dot(o2)
print(o4)




