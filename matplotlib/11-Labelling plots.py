import matplotlib.pyplot as plt
import numpy as np

mu, sigma = 115, 15

x = mu + sigma * np.random.randn(10000)

fig, ax = plt.subplots(figsize=(5, 2.7), layout="constrained")

# 绘制直方图
# ax.hist(x, 50)：绘制 x 数据的直方图，分为 50 个 bins（区间）。
# density=True：将直方图的 y 轴值归一化，使其成为概率密度，而非频数。
# facecolor="C0"：设置直方图柱子的填充颜色。
# alpha=0.75：设置柱子的透明度为 0.75。
n, bins, patches = ax.hist(x, 50, density=True, facecolor="C0", alpha=0.75)

ax.set_xlabel("Length [cm]")
ax.set_ylabel("Probability")

ax.set_title("Aardvark lengths\n (not really)")

ax.text(75, 0.025, r"$\mu=115,\ \sigma=15$")

ax.axis([55, 175, 0, 0.03])

ax.grid(True)

plt.show()
