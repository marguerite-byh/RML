
from pykrige.uk import UniversalKriging
import time
import cv2
import tensorly as tl
import numpy as np
import datetime
import matplotlib.pyplot as plt
start_time = time.time()

def generate_aggregated_signal_with_noise(N, M, K, cluster_size_range, snr):
    x_matrices = []
    centers = []
    center_values = []
    for _ in range(K):
        x = np.zeros((N, M))
        center_x = np.random.randint(0, N)
        center_y = np.random.randint(0, M)
        cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1])
        centers.append((center_x, center_y))
        sigma = cluster_size / 3
        for i in range(N):
            for j in range(M):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                if distance == 0:
                    x[i, j] += np.exp(-1 / (2 * sigma ** 2))
                else:
                    x[i, j] += np.exp(-distance ** 2 / (2 * sigma ** 2))

        x_matrices.append(x)
        center_values.append(np.mean(x ** 2))

    max_center_value = max(center_values)
    noise_power = max_center_value / (10 ** (snr / 10))
    aggregated_x = np.sum(x_matrices, axis=0)
    noise = np.abs(np.sqrt(noise_power) * np.random.randn(N, M))
    aggregated_x += noise

    return x_matrices, aggregated_x
N, M, K = 100, 100, 3
cluster_size_range = (5, 25)
SNR = 20  # dB
sample_rate = 0.15
x_matrices, aggregated_x = generate_aggregated_signal_with_noise(N, M, K, cluster_size_range, SNR)

# 假设 aggregated_x 已经生成
import matplotlib.pyplot as plt
from skimage.transform import resize

# 假设 aggregated_x 已经生成
# 缩放 aggregated_x 到 360x360
resized_aggregated_x = resize(aggregated_x, (360, 360), anti_aliasing=True)

# 绘制图像
plt.imshow(resized_aggregated_x, cmap='jet', origin='lower')

# 关闭坐标轴
plt.axis('off')

# 保存图像，去除所有边距
plt.savefig('aggregated_signal.png', dpi=300, bbox_inches='tight', pad_inches=0)
from PIL import Image

# 读取图像
image_path = "test.jpg"
output_path = "test.jpg"
scale_factor = 0.1  # 缩小 50%

# 打开图像
with Image.open(image_path) as img:
    # 计算新的尺寸
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)

    # 调整大小
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 保存图像
    img_resized.save(output_path)

print(f"图像已按比例缩小并保存到 {output_path}")