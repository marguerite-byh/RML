import numpy as np
import matplotlib.pyplot as plt
from pykrige.uk import UniversalKriging
import time

start_time = time.time()

def generate_aggregated_signal(N, M, K, cluster_size_range):
    """
    生成具有团聚特点的二维稀疏信号
    """
    x = np.zeros((N, M))
    for _ in range(K):
        center_x = np.random.randint(0, N)
        center_y = np.random.randint(0, M)
        cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1])
        sigma = cluster_size / 3
        for i in range(N):
            for j in range(M):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                x[i, j] += np.exp(-distance ** 2 / (2 * sigma ** 2))
    return x

# 生成 n*m 的矩阵
n, m = 100, 100  # 降低分辨率
N, M, K = 100, 100, 3
cluster_size_range = (10, 20)
original_matrix = generate_aggregated_signal(N, M, K, cluster_size_range)

# 随机采样，设置空缺值为0
sample_matrix = np.copy(original_matrix)
mask = np.random.choice([True, False], size=original_matrix.shape, p=[0.07, 0.93])
sample_matrix[~mask] = 0

# 将空缺值设为 np.nan
data_sta = np.where(sample_matrix == 0, np.nan, sample_matrix)
lon_sta, lat_sta = np.meshgrid(np.arange(m), np.arange(n))
lon_sta, lat_sta = lon_sta.flatten().astype(float), lat_sta.flatten().astype(float)

# 获取有效的采样点
valid_idx = ~np.isnan(data_sta.flatten())
lon_valid = lon_sta[valid_idx]
lat_valid = lat_sta[valid_idx]
data_valid = data_sta.flatten()[valid_idx]

# 降采样减少数据点
sample_size = min(500, len(lon_valid))
indices = np.random.choice(len(lon_valid), sample_size, replace=False)
lon_valid_sampled = lon_valid[indices]
lat_valid_sampled = lat_valid[indices]
data_valid_sampled = data_valid[indices]

# 克里金插值
UK = UniversalKriging(
    lon_valid_sampled, lat_valid_sampled, data_valid_sampled,
    variogram_model='hole-effect',
    verbose=False,
    enable_plotting=False
)

# 生成插值网格
lon2D, lat2D = np.meshgrid(np.arange(m).astype(float), np.arange(n).astype(float))
z2, ss2 = UK.execute('grid', np.arange(m).astype(float), np.arange(n).astype(float))

# 绘制结果
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.title('Original Signal')
plt.imshow(original_matrix, cmap='jet', origin='lower')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Observed Image")
plt.imshow(sample_matrix, cmap='jet', origin='lower')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title('Recovered Signal')
plt.imshow(z2, cmap='jet', origin='lower')
plt.colorbar()

plt.tight_layout()
plt.show()

# 计算 RMSE
rmse = np.sqrt(np.mean((z2 - original_matrix) ** 2))
print(f"原始信号与恢复信号之间的均方根误差 (RMSE): {rmse}")

end_time = time.time()
# 计算执行时间
execution_time = end_time - start_time
# 输出执行时间
print("程序执行时间：", execution_time, "秒")
