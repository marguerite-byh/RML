import numpy as np
import matplotlib.pyplot as plt

# 定义空间网格大小
grid_size = 100
x = np.linspace(0, 100, grid_size)
y = np.linspace(0, 100, grid_size)
X, Y = np.meshgrid(x, y)

# 定义发射源的位置和功率
sources = [
    {'position': (0, 0), 'power': 100},
    {'position': (20, 20), 'power': 50}
]

# 计算每个网格点的频谱强度
intensity = np.zeros((grid_size, grid_size))

for source in sources:
    sx, sy = source['position']
    power = source['power']
    distance = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2)
    # 避免除以零的情况
    distance[distance <1 ] = 1
    intensity += power / (4 * np.pi * distance ** 2)

# 绘制频谱强度图
plt.imshow(intensity, extent=(-50, 50, -50, 50), origin='lower', cmap='jet')
plt.colorbar(label='Intensity')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Spectrum Intensity Map')
plt.show()
