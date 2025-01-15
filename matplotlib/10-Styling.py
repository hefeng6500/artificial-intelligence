import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots(figsize=(5, 2.7))

data1 = np.random.randint(0, 16, size=100)
data2 = np.random.randint(-16, 0, size=100)

x = np.arange(len(data1))

ax.plot(x, np.cumsum(data1), color="blue", linewidth=3, linestyle="--")

(l,) = ax.plot(x, np.cumsum(data2), color="orange", linewidth=2)

l.set_linestyle(":")

plt.show()
