import matplotlib.pyplot as plt
import numpy as np

# 设置随机数生成器的种子，这样每次运行代码时生成的随机数序列都是相同的。
np.random.seed(19680801)  # seed the random number generator.

# data 是一个字典，包含四个键值对：
#   "a"：一个从0到49的数组。
#   "c"：一个包含50个随机整数的数组，范围在0到50之间。
#   "d"：一个包含50个标准正态分布随机数的数组。
#   "b"：通过 "a" 加上10倍的随机正态分布数生成。
#   "d"：取 "d" 的绝对值并乘以100。

data = {"a": np.arange(50), "c": np.random.randint(0, 50, 50), "d": np.random.randn(50)}
data["b"] = data["a"] + 10 * np.random.randn(50)
data["d"] = np.abs(data["d"]) * 100

# 设置图表的大小为宽5英寸，高2.7英寸。
# 使用约束布局，确保图表元素不会重叠
fig, ax = plt.subplots(figsize=(5, 2.7), layout="constrained")

# ax.scatter 在子图上绘制散点图。
#   "a" 和 "b" 是数据中的键，分别作为x轴和y轴的数据。
#   c="c" 使用 "c" 键对应的值作为散点的颜色。
#   s="d" 使用 "d" 键对应的值作为散点的大小。
#   data=data 指定数据来源为 data 字典。
ax.scatter("a", "b", c="c", s="d", data=data)

# 设置x轴和y轴的标签
ax.set_xlabel("entry a")
ax.set_ylabel("entry b")

plt.show()
