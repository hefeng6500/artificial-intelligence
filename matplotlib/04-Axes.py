import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x = np.linspace(0, 4 * np.pi, 200)
y = np.sin(x)

ax.plot(x, y)

# 设置标题
ax.set_title(
    "A simple chirp",
    size=20,
    color="#375E97",
    weight="bold",
    alpha=0.8,
    family="microsoft yahei",
    style="normal",
    variant="small-caps",
    stretch="ultra-condensed",
    rotation=0,
)

# 设置坐标轴标签
ax.set_xlabel("Time ($\pi$ s)")
ax.set_ylabel("Amplitude")


# 将图表底部的边框（spine）的位置设置为与x轴的数据值为0的位置对齐
ax.spines["bottom"].set_position(("data", 0))

plt.show()
