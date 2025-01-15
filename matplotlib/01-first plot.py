import matplotlib.pyplot as plt
import numpy as np

# pyplot模块提供了一个类似于MATLAB的绘图系统，称为“绘图系统”。

# 生成一个等差数列,从0开始，到2 * np.pi（即2π，圆周率π的两倍）结束，总共包含200个元素
x = np.linspace(0, 2 * np.pi, 200)
# 计算x数组中每个元素的正弦值
y = np.sin(x)

# 创建一个图形(figure)和一个子图(axes)
fig, ax = plt.subplots()
# ax对象的plot方法绘制x和y的图形
ax.plot(x, y)
# 显示图形
plt.show()
