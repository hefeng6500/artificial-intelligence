import numpy as np

a = np.array([[0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 11]])

b = a

# b 和 a 指向同一个对象
print(b is a)

print(id(a))

c = a.view()
print(c is a)  # False
print(c.base is a)  # True 浅拷贝

c = c.reshape((2, 6))
print(c)
print(a.shape)

c[0, 4] = 1234
print(a)

s= a[:, 1:3] # 所有的行，第1列到第3列（不包括第3列）
s[:] = 10
print(s)
print(a)

# 深拷贝
d = a.copy()
print(d is a)  # False
d[0, 0] = 9999
print(a)