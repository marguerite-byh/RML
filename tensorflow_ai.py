import numpy as np
from pykrige.ok import OrdinaryKriging
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 生成带噪声的频谱数据
def generate_aggregated_signal_with_noise(N, M, K, cluster_size_range, snr):
    x_matrices = []
    for _ in range(K):
        x = np.zeros((N, M))
        center_x, center_y = np.random.randint(0, N), np.random.randint(0, M)
        cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1])
        sigma = cluster_size / 3

        for i in range(N):
            for j in range(M):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                value = np.exp(-distance ** 2 / (2 * sigma ** 2))
                x[i, j] += value

        x_matrices.append(x)

    aggregated_x = np.sum(x_matrices, axis=0)
    noise_power = np.mean(aggregated_x ** 2) / (10 ** (snr / 10))
    noise = np.sqrt(noise_power) * np.random.randn(N, M)
    aggregated_x += noise
    return aggregated_x

# 克里金插值补全
def kriging_completion(data_with_missing):
    N, M = data_with_missing.shape
    x, y = np.arange(N).astype(float), np.arange(M).astype(float)  # 将整型转换为浮点型
    X, Y = np.meshgrid(x, y, indexing="ij")

    # 提取非缺失值的点和对应的值
    points = np.array([(i, j) for i in range(N) for j in range(M) if not np.isnan(data_with_missing[i, j])])
    values = np.array([data_with_missing[i, j] for i, j in points])

    # 克里金插值
    ok = OrdinaryKriging(points[:, 0], points[:, 1], values, variogram_model="linear", verbose=False,
                         enable_plotting=False)
    z, _ = ok.execute("grid", x, y)  # 使用浮点网格点
    return z


# 张量补全模型
def tensor_completion_model(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same"))
    model.compile(optimizer="adam", loss="mse")
    return model

# 主流程
N, M, K = 100, 100, 5
cluster_size_range = (5, 15)
snr = 20
original_data = generate_aggregated_signal_with_noise(N, M, K, cluster_size_range, snr)

# 随机丢失50%的数据
mask = np.random.rand(N, M) < 0.5
data_with_missing = np.copy(original_data)
data_with_missing[mask] = np.nan

# 使用克里金补全
kriging_result = kriging_completion(data_with_missing)

# 神经网络优化
data_input = np.expand_dims(kriging_result, axis=-1)
data_input = np.expand_dims(data_input, axis=0)  # 批处理维度
model = tensor_completion_model((N, M, 1))
model.fit(data_input, data_input, epochs=10, batch_size=1, verbose=1)
nn_result = model.predict(data_input).squeeze()

# 对比补全结果
mse_kriging = np.mean((original_data - kriging_result) ** 2)
mse_nn = np.mean((original_data - nn_result) ** 2)
print(f"克里金补全 MSE: {mse_kriging:.4f}")
print(f"神经网络优化后 MSE: {mse_nn:.4f}")

# 可视化结果
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.title("Original Data")
plt.imshow(original_data, cmap="hot")
plt.colorbar()

plt.subplot(1, 4, 2)
plt.title("Data with Missing")
plt.imshow(np.where(np.isnan(data_with_missing), 0, data_with_missing), cmap="hot")
plt.colorbar()

plt.subplot(1, 4, 3)
plt.title("Kriging Completion")
plt.imshow(kriging_result, cmap="hot")
plt.colorbar()

plt.subplot(1, 4, 4)
plt.title("NN Optimized Completion")
plt.imshow(nn_result, cmap="hot")
plt.colorbar()
plt.show()
