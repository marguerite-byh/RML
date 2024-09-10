import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth
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
def normalize(signal):
    """
    归一化信号到 [0, 1] 范围

    参数：
    signal - 需要归一化的信号

    返回：
    归一化后的信号
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    return (signal - min_val) / (max_val - min_val)

# 测试生成信号并绘制热力图
N, M, K = 100, 100, 2
cluster_size_range = (10, 20)
x = generate_aggregated_signal(N, M, K, cluster_size_range)
n1, n2 = x.shape
# 采样率
sample_rate = 0.3
P = np.zeros(n1 * n2)  # 初始化为一维数组
MM = x.flatten()
pos = np.sort(np.random.choice(n1 * n2, int(n1 * n2 * sample_rate), replace=False))
P[pos] = MM[pos]  # 将观测值赋给 P
index1 = np.where(P)[0]
P[index1] = 1
P = P.reshape(n1, n2)  # 重塑为矩阵
print(P)
Z=x*P
C=int(n1 * n2 * sample_rate)
def cs_ista(y, A, lambda_val=2e-5, epsilon=1e-5, itermax=20000):
    """
    迭代软阈值算法 (ISTA)

    参数：
    y         - 测量向量
    A         - 测量矩阵
    lambda_val - 噪声情况下的去噪参数
    epsilon   - 误差阈值
    itermax   - 最大迭代次数

    返回：
    x_hat     - 最后一次估计值
    error     - 重建误差
    """
    N = A.shape[1]  # A的列数，即信号的长度
    error = []  # 存储每次迭代的误差
    x_1 = np.zeros(N)  # 初始化 x_1

    for i in range(itermax):
        # 计算梯度
        g_1 = A.T @ (y - A @ x_1)

        # 步长
        alpha = 1

        # 计算 x_2
        x_2 = x_1 + alpha * g_1

        # 软阈值处理
        x_hat = np.sign(x_2) * np.maximum(np.abs(x_2) - alpha * lambda_val, 0)

        # 计算误差
        error_1 = np.linalg.norm(x_hat - x_1) / np.linalg.norm(x_hat)
        error_2 = np.linalg.norm(y - A @ x_hat)
        error.append([error_1, error_2])

        # 判断是否满足误差阈值条件
        if error_1 < epsilon or error_2 < epsilon:
            break
        else:
            # 更新变量
            x_1 = x_hat

    return x_hat, np.array(error)




# 生成测量矩阵
A = np.random.randn(C, N * M)
A = (1 / np.sqrt(C)) * A  # 归一化
A = orth(A.T).T  # 正交化

# 生成测量向量
y = A @Z.flatten()  # + e (这里没有添加噪声)

# 调用ISTA算法恢复信号
x_rec_flat, error1 = cs_ista(y, A, lambda_val=5e-3, epsilon=3e-5, itermax=10000)
R = x_rec_flat.reshape(N, M)
x_normalized = normalize(x)
x_rec_normalized = normalize(R)

# 绘制原始信号和恢复信号的热力图

# plt.subplot(1, 2, 1)
# plt.imshow(x_normalized, cmap='jet', aspect='auto')
# plt.title('Original Signal')
# plt.colorbar()


plt.subplot(1, 2, 1)
plt.imshow(Z, cmap='jet', aspect='auto')
plt.title("Oberseved Image")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(x_rec_normalized, cmap='jet', aspect='auto')
plt.title('Recovered Signal')
plt.colorbar()

plt.tight_layout()
plt.show()

# 计算并输出均方根误差
rmse = np.sqrt(np.mean((x -Z) ** 2))
print(f"原始信号与恢复信号之间的均方根误差 (RMSE): {rmse}")
