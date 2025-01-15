import matplotlib.pylab as plt
import numpy as np

fig, ax = plt.subplots()  # 创建一个图形和坐标轴对象
data = np.random.rand(10) * 100  # 生成10个随机数，范围为0到100

print(data)

ax.plot(data)  # 绘制折线图
ax.scatter([1, 2, 3, 4], [50, 20, 10, 70])  # 绘制散点图
ax.bar([1, 2, 3, 4], [70, 20, 50, 90])  # 绘制柱状图


ax.set_xlabel("X Label")  # 设置x轴标签
ax.set_ylabel("Y Label")  # 设置y轴标签

ax.set_title("Simple Plot")  # 设置标题
ax.legend(["Line", "Scatter", "Bar"])  # 设置图例
ax.grid(True)  # 显示网格线

plt.show()
