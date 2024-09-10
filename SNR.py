import numpy as np
N, M, K = 100, 100, 3
cluster_size_range = (5, 15)
SNR=10 #dB
x_matrices = []
centers = []
center_values = []
for _ in range(K):
        x = np.zeros((N, M))

        # for _ in range(K):
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

        # 分别计算每个簇的能量
        x_matrices.append(x)
        center_values.append(np.max(x**2))
max_center_value = max(center_values)
signal_power = np.mean(x ** 2)
    # 计算噪声功率，基于信噪比
noise_power = max_center_value / (10 ** (SNR / 10))
    # 将所有信号矩阵叠加在一起组成一个新的信号矩阵
aggregated_x = np.sum(x_matrices, axis=0)
    # 生成背景噪声矩阵
noise = np.sqrt(noise_power) * np.random.randn(N, M)
    # 将噪声加入信号
aggregated_x += noise
#打印x_matrices的形状
print(np.array(x_matrices).shape)
def generate_aggregated_signal_with_noise_rbf(N, M, K, cluster_size_range, snr):
    # 生成初始信号矩阵
    x = np.zeros((N, M))
    centers=[]
    for _ in range(K):
        center_x = np.random.randint(0, N)
        center_y = np.random.randint(0, M)
        cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1])
        centers.append((center_x,center_y))
        sigma = cluster_size / 3
        for i in range(N):
            for j in range(M):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                x[i, j] += np.exp(-distance ** 2 / (2 * sigma ** 2))
        # 计算每个簇中心对应的信号值
    center_values = [x[center_x, center_y] for center_x, center_y in centers]

    # 找出簇中心信号值的最大值
    max_center_value = max(center_values)
    # 计算信号功率
    signal_power = np.mean(max_center_value ** 2)
    # 计算噪声功率，基于信噪比
    noise_power = signal_power / (10 ** (snr / 10))
    # 生成背景噪声矩阵
    noise = np.sqrt(noise_power) * np.random.randn(N, M)
    # 将噪声加入信号
    x += noise
    return x