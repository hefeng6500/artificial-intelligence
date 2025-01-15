import matplotlib.pyplot as plt
import numpy as np

# fig = plt.figure()  # an empty figure with no Axes

# fig = plt.subplot()  # a figure with a single Axes

# fig, axs = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes

# a figure with one Axes on the left, and two on the right:
fig, axs = plt.subplot_mosaic([["left", "right_top"], ["left", "right_bottom"]])

# 布局被分为两行，每行有两个区域
# "left"（与第一行的"left"共享垂直空间）和 "right_bottom"

plt.show()
