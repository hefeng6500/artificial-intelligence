import matplotlib.pyplot as plt
import numpy as np

# 创建图表
fig, ax = plt.subplots()

# 数据
x = np.linspace(0, 3 * np.pi, 200)  # 可以动态修改横轴范围
y = np.sin(x)

# 设置 X 轴范围为 0 到 3π，Y 轴范围为 -1 到 1
ax.set_xlim(0, 3 * np.pi)
ax.set_ylim(-1.2, 1.2)


# 绘制曲线
ax.plot(x, y)

# 设置轴位置
ax.spines["bottom"].set_position(("data", 0))

# 动态生成 X 轴刻度和标签
x_min, x_max = ax.get_xlim()  # 获取 X 轴范围
step = np.pi / 2  # 每 1/2π 为一个刻度
xticks = np.arange(0, x_max + step, step)  # 从 0 到 x_max，每 step 生成一个刻度
xtick_labels = [
    (
        r"$0$"
        if t == 0
        else (
            r"$\frac{" + str(int(2 * t / np.pi)) + r"}{2}\pi$"
            if t % np.pi != 0
            else r"$" + str(int(t / np.pi)) + r"\pi$"
        )
    )
    for t in xticks
]

ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels)

# 显示图表
plt.show()
