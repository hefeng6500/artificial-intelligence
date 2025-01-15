import matplotlib.pyplot as plt
import numpy as np


def my_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph.
    """
    out = ax.plot(data1, data2, **param_dict)  # ** 是字典解包
    return out


data1, data2, data3, data4 = np.random.randn(4, 100)  # make 4 random data sets

# 创建一个 figure, 一行两列, 大小为 5*2.7
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.7))

my_plotter(ax1, data1, data2, {"marker": "x", "color": "r"})
my_plotter(ax2, data3, data4, {"marker": "o"})

plt.show()
