import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import ConciseDateFormatter

fig, ax = plt.subplots(figsize=(5, 2.7), layout="constrained")

dates = np.arange(
    np.datetime64("2021-11-15"), np.datetime64("2021-12-25"), np.timedelta64(1, "h")
)

# np.random.randn(len(dates))：生成与日期数组相同长度的标准正态分布的随机数。
# np.cumsum()：计算这些随机数的累积和，生成一个随机游走数据序列。
data = np.cumsum(np.random.randn(len(dates)))

ax.plot(dates, data)

# ax.xaxis.get_major_locator()：获取 x 轴的主要定位器（主要刻度位置）。
# ConciseDateFormatter()：使用简洁的日期格式化器，自动根据时间间隔调整日期显示格式（例如：显示为“Nov 15”而非完整日期）。
ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))

plt.show()
