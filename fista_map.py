import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth


def cs_ista(y, A, lambda_val=2e-5, epsilon=1e-4, itermax=10000):
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

# 参数设定
N = 100  # 信号的行数
M = 100  # 信号的列数
P = 765  # 测量数量
K = 2  # 团聚点的数量
cluster_size_range = (20, 40)  # 每个团聚点的大小范围

# 生成具有团聚特点的二维稀疏信号
x = generate_aggregated_signal(N, M, K, cluster_size_range)

# 生成测量矩阵
A = np.random.randn(P, N * M)
A = (1 / np.sqrt(P)) * A  # 归一化
A = orth(A.T).T  # 正交化

# 生成测量向量
y = A @ x.flatten()  # + e (这里没有添加噪声)

# 调用ISTA算法恢复信号
x_rec_flat, error1 = cs_ista(y, A, lambda_val=5e-3, epsilon=1e-4, itermax=5000)
x_rec = x_rec_flat.reshape(N, M)

# 绘制原始信号和恢复信号的热力图
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.contourf(x, cmap='jet')
plt.title('原始信号')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.contourf(x_rec, cmap='jet')
plt.title('恢复信号')
plt.colorbar()

plt.tight_layout()
plt.show()

# 计算并输出均方根误差
rmse = np.sqrt(np.mean((x - x_rec) ** 2))
print(f"原始信号与恢复信号之间的均方根误差 (RMSE): {rmse}")
