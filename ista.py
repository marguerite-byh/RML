import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf


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


# 生成 n*m 的矩阵
N, M, K = 100, 100, 3
cluster_size_range = (5, 25)
SNR = 10  # dB
sample_rate = 0.05
x_matrices, aggregated_x = generate_aggregated_signal_with_noise(N, M, K, cluster_size_range, SNR)



# 随机采样，设置空缺值为0

sample_matrix = np.copy(aggregated_x)
mask = np.random.choice([True, False], size=aggregated_x.shape, p=[sample_rate, 1 - sample_rate])
sample_matrix[~mask] = 0

# 将空缺值设为 np.nan
data_sta = np.where(sample_matrix == 0, np.nan, sample_matrix)
lon_sta, lat_sta = np.meshgrid(np.arange(M), np.arange(N))
lon_sta, lat_sta = lon_sta.flatten().astype(float), lat_sta.flatten().astype(float)

# 获取有效的采样点
valid_idx = ~np.isnan(data_sta.flatten())
lon_valid = lon_sta[valid_idx]
lat_valid = lat_sta[valid_idx]
data_valid = data_sta.flatten()[valid_idx]

# 使用 RBF 进行插值
rbf = Rbf(lon_valid, lat_valid, data_valid, function='multiquadric')

# 生成插值网格
lon2D, lat2D = np.meshgrid(np.arange(M).astype(float), np.arange(N).astype(float))
z = rbf(lon2D.flatten(), lat2D.flatten()).reshape((N, M))

# # 显示原始信号
# plt.subplot(1, 3, 1)
# plt.title('Original Signal')
# plt.imshow(aggregated_x, cmap='jet', origin='lower')
# plt.colorbar()

# 绘制每个矩阵及其相加后的矩阵
plt.figure(figsize=(14, 6))
# 添加总标题
plt.suptitle(f'SNR={SNR}dB, Sample Rate={sample_rate}')
# 绘制每个矩阵
for i, x in enumerate(x_matrices):
    plt.subplot(2, K + 1, i + 1)
    plt.title(f'Matrix {i + 1}')
    plt.imshow(x, cmap='jet', origin='lower')
    plt.colorbar()

# 绘制相加后的矩阵
plt.subplot(2, K + 1, K + 2)
plt.title('Aggregated Matrix')
plt.imshow(aggregated_x, cmap='jet', origin='lower')
plt.colorbar()

# 显示观测图像
plt.subplot(2, K+1, K+3)
plt.title("Observed Image")
plt.imshow(sample_matrix, cmap='jet', origin='lower')
plt.colorbar()

# 显示恢复信号
plt.subplot(2, K+1, K+4)
plt.title('Recovered Signal')
plt.imshow(z, cmap='jet', origin='lower')
plt.colorbar()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# # 调整布局
# plt.tight_layout(rect=[0, 0, 1, 0.95])

# 计算 RMSE 和 Relative Error
try:
    rmse = np.sqrt(np.mean((z - aggregated_x) ** 2))
    relative_error = np.linalg.norm(z - aggregated_x) / np.linalg.norm(aggregated_x)

    # 打印结果
    plt.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nRelative Error: {relative_error:.4f}',
             transform=plt.gcf().transFigure,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5))

    print(f"Relative Error: {relative_error}")
    print(f"原始信号与恢复信号之间的均方根误差 (RMSE): {rmse}")
except Exception as e:
    print(f"计算过程中出现错误：{e}")

plt.show()
