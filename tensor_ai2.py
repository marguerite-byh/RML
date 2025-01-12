import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from sklearn.metrics import mean_squared_error # 使用多模式点积来重建张量
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.tenalg import multi_mode_dot
import tensorflow as tf
from tensorflow.keras import layers, models


# 进行张量分解并重构张量
def tensor_completion(tensor_missing, rank=5, max_iter=100):
    # 填充缺失的值（此处将nan填充为0或其他值）
    tensor_missing_filled = tl.fill_missing(tensor_missing, fill_value=0)

    # 使用 parafac 进行张量分解
    factors = parafac(tensor_missing_filled, rank=rank, init='random', tol=1e-6)

    # 通过因子矩阵重构张量
    reconstructed_tensor = multi_mode_dot(factors, modes=[0, 1, 2])  # 针对三维张量的每个模式进行点积

    return reconstructed_tensor


# 随机丢失数据（生成缺失数据）
def create_missing_data(tensor, missing_rate=0.5):
    # 随机生成缺失数据的掩码
    mask = np.random.rand(*tensor.shape) < missing_rate
    tensor_with_missing = tensor.copy()
    tensor_with_missing[mask] = np.nan  # 将数据设置为 NaN
    return tensor_with_missing, mask

# 张量分解补全

# 生成带噪声的频谱数据
def generate_aggregated_signal_with_noise(N, M, K, cluster_size_range, snr):
    x_matrices = []
    centers = []
    center_values = []
    for _ in range(K):
        x = np.zeros((N, M))
        center_x, center_y = np.random.randint(0, N), np.random.randint(0, M)
        cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1])
        centers.append((center_x, center_y))
        sigma = cluster_size / 3

        for i in range(N):
            for j in range(M):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                value = np.exp(-distance ** 2 / (2 * sigma ** 2))
                x[i, j] += value

        x_matrices.append(x)
        center_values.append(np.mean(x ** 2))

    max_center_value = max(center_values)
    noise_power = max_center_value / (10 ** (snr / 10))

    aggregated_x = np.sum(x_matrices, axis=0)
    noise = np.sqrt(noise_power) * np.random.randn(N, M)
    aggregated_x += noise

    return x_matrices, aggregated_x,centers

# 模拟频点在时间维度上的移动
def generate_time_continuous_spectrum(N, M, T, K, cluster_size_range, snr, max_shift=1):
    spectra = []
    _, initial_spectrum, centers = generate_aggregated_signal_with_noise(N, M, K, cluster_size_range, snr)
    spectra.append(initial_spectrum)

    for t in range(1, T):
        new_spectrum = np.zeros((N, M))
        new_centers = []

        for center in centers:
            # 模拟频点的随机移动
            shift_x = np.random.randint(-max_shift, max_shift + 1)
            shift_y = np.random.randint(-max_shift, max_shift + 1)
            new_center_x = np.clip(center[0] + shift_x, 0, N - 1)
            new_center_y = np.clip(center[1] + shift_y, 0, M - 1)
            new_centers.append((new_center_x, new_center_y))

            # 重新生成每个频点的高斯分布
            sigma = cluster_size_range[1] / 3  # 使用最大集群尺寸
            for i in range(N):
                for j in range(M):
                    distance = np.sqrt((i - new_center_x) ** 2 + (j - new_center_y) ** 2)
                    value = np.exp(-distance ** 2 / (2 * sigma ** 2))
                    new_spectrum[i, j] += value

        # 添加噪声
        max_value = np.max(new_spectrum)
        noise_power = max_value / (10 ** (snr / 10))
        noise = np.sqrt(noise_power) * np.random.randn(N, M)
        new_spectrum += noise

        spectra.append(new_spectrum)
        centers = new_centers

    return np.array(spectra)

def build_autoencoder(input_shape):
    # 构建卷积自编码器
    model = models.Sequential()

    # 编码器
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    # 解码器
    model.add(layers.Conv3DTranspose(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling3D(size=(2, 2, 2)))
    model.add(layers.Conv3DTranspose(1, (3, 3, 3), activation='sigmoid', padding='same'))

    model.compile(optimizer='adam', loss='mse')
    return model

# 使用自编码器进行缺失数据的补全
def neural_network_completion(tensor_missing, epochs=10, batch_size=8):
    input_data = np.expand_dims(tensor_missing, axis=-1)  # 为卷积网络添加通道维度

    # 构建并训练模型
    model = build_autoencoder(input_data.shape[1:])
    model.fit(input_data, input_data, epochs=epochs, batch_size=batch_size)

    # 补全数据
    completed_tensor = model.predict(input_data)
    completed_tensor = completed_tensor.squeeze()  # 移除多余的维度
    return completed_tensor
# 主流程
# 设置参数
N, M, T = 100, 100, 30  # 空间尺寸 (N, M) 和时间步数 T
K = 5  # 信号源数量
cluster_size_range = (5, 15)
snr = 20
max_shift = 2  # 每步的最大移动范围

# 生成时间连续的频谱图
time_continuous_spectra = generate_time_continuous_spectrum(N, M, T, K, cluster_size_range, snr, max_shift)

# 创建缺失数据
time_continuous_spectra_missing, mask = create_missing_data(time_continuous_spectra, missing_rate=0.5)

# 使用张量低秩补全
completed_tensor_parafac = tensor_completion(time_continuous_spectra_missing, rank=5, max_iter=100)

# 使用神经网络补全
completed_tensor_nn = neural_network_completion(time_continuous_spectra_missing, epochs=10, batch_size=8)

# 可视化补全结果
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# 原始数据（取部分时间步进行显示）
for t in range(3):
    axs[0, t].imshow(time_continuous_spectra[t], cmap='hot', aspect='auto')
    axs[0, t].set_title(f"Original - Time Step {t}")
    axs[0, t].colorbar()

# 补全后的数据（PARAFAC）
for t in range(3):
    axs[1, t].imshow(completed_tensor_parafac[t], cmap='hot', aspect='auto')
    axs[1, t].set_title(f"Completed - PARAFAC Time Step {t}")
    axs[1, t].colorbar()

plt.tight_layout()
plt.show()

# 计算与原始数据的均方误差（MSE）
mse_parafac = mean_squared_error(time_continuous_spectra[~np.isnan(time_continuous_spectra)], completed_tensor_parafac[~np.isnan(time_continuous_spectra)])
mse_nn = mean_squared_error(time_continuous_spectra[~np.isnan(time_continuous_spectra)], completed_tensor_nn[~np.isnan(time_continuous_spectra)])

print(f"PARAFAC MSE: {mse_parafac:.4f}")
print(f"Neural Network MSE: {mse_nn:.4f}")
