import numpy as np
import matplotlib.pyplot as plt


def euclidean_dist(x1, y1, x2, y2):
    """
    计算二维平面上两点之间的欧几里得距离
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def haversine_dist(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度点之间的哈弗赛公式距离
    """
    R = 6371  # 地球半径，单位为公里
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance


def generate_aggregated_signal(N, M, K, cluster_size_range):
    """
    生成具有团聚特点的二维稀疏信号

    参数：
    N                 - 信号的行数
    M                 - 信号的列数
    K                 - 团聚点的数量
    cluster_size_range - 每个团聚点的大小范围（元组）

    返回：
    x                 - 生成的二维稀疏信号
    """
    x = np.zeros((N, M))
    for _ in range(K):
        center_x = np.random.randint(0, N)
        center_y = np.random.randint(0, M)
        cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1])

        # 创建一个高斯分布
        sigma = cluster_size / 3
        for i in range(N):
            for j in range(M):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                x[i, j] += np.exp(-distance ** 2 / (2 * sigma ** 2))

    return x


def interp_IDW(lon_sta, lat_sta, data_sta, lon2D, lat2D):
    n_sta = len(lon_sta)
    ny, nx = np.shape(lon2D)
    data2D = np.zeros((ny, nx))

    # 将站点经纬度和数据转换为数组
    lon_sta = np.array(lon_sta)
    lat_sta = np.array(lat_sta)
    data_sta = np.array(data_sta)

    for j in range(ny):
        for i in range(nx):  # 遍历二维每一个格点
            # 计算当前格点与所有站点的距离
            dist = np.sqrt((lon_sta - lon2D[j, i]) ** 2 + (lat_sta - lat2D[j, i]) ** 2)
            dist = np.maximum(dist, 1.0)  # 避免除以零

            # 计算权重
            wgt = 1.0 / np.power(dist, 2)
            wgt_sum = np.sum(wgt[~np.isnan(data_sta)])
            arg_sum = np.sum(wgt[~np.isnan(data_sta)] * data_sta[~np.isnan(data_sta)])

            # 计算插值结果
            if wgt_sum > 0:
                data2D[j, i] = arg_sum / wgt_sum
            else:
                data2D[j, i] = np.nan  # 如果没有有效数据，填充空缺值

    return data2D


# 生成 n*m 的矩阵
n, m = 100, 100
N, M, K = 100, 100, 3
cluster_size_range = (10, 20)
original_matrix = generate_aggregated_signal(N, M, K, cluster_size_range)

# 随机采样，设置空缺值为0
sample_matrix = np.copy(original_matrix)
mask = np.random.choice([True, False], size=original_matrix.shape, p=[0.1, 0.9])
sample_matrix[~mask] = 0

# 将空缺值设为 np.nan
data_sta = np.where(sample_matrix == 0, np.nan, sample_matrix).flatten()
lon_sta, lat_sta = np.meshgrid(np.arange(m), np.arange(n))
lon_sta, lat_sta = lon_sta.flatten(), lat_sta.flatten()

# 生成插值网格
lon2D, lat2D = np.meshgrid(np.arange(m), np.arange(n))

# 进行插值
result = interp_IDW(lon_sta, lat_sta, data_sta, lon2D, lat2D)

# 绘制结果
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.title('Original Signal')
plt.imshow(original_matrix, cmap='jet', origin='lower')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Oberseved Image")
plt.imshow(sample_matrix, cmap='jet', origin='lower')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title('Recovered Signal')
plt.imshow(result, cmap='jet', origin='lower')
plt.colorbar()

plt.tight_layout()
plt.show()
rmse = np.sqrt(np.mean((result -original_matrix) ** 2))
print(f"原始信号与恢复信号之间的均方根误差 (RMSE): {rmse}")