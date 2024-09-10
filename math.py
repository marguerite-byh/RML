#绘制两个函数图像np.exp(-d ** 2 / (2 * sigma ** 2))和1/d**2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
x = np.linspace(0, 10, 100)
y1 = np.exp(-x ** 2 / 2)
y2 = 1 / x ** 2
plt.plot(x, y1, label='exp(-x^2/2)')
plt.plot(x, y2, label='1/x^2')
plt.legend()
plt.show()
# Compare this snippet from
