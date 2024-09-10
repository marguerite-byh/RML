import numpy as np
from scipy.linalg import orth
import matplotlib.pyplot as plt


def cs_fista(y, A, lambda_val=2e-5, epsilon=1e-4, itermax=10000):
    """
    快速迭代软阈值算法 (FISTA)

    参数：
    y         - 测量向量
    A         - 测量矩阵
    lambda_val - 噪声情况下的去噪参数
    epsilon   - 误差阈值
    itermax   - 最大迭代次数

    返回：
    x_2       - 最后一次估计值
    error     - 重建误差
    """
    # 初始化
    N = A.shape[1]  # A的列数，即信号的长度
    error = []  # 存储每次迭代的误差

    x_0 = np.zeros(N)  # 初始化 x_0
    x_1 = np.zeros(N)  # 初始化 x_1
    t_0 = 1  # 初始化 t_0

    for i in range(itermax):
        # 计算 t_1
        t_1 = (1 + np.sqrt(1 + 4 * t_0 ** 2)) / 2

        # 计算 z_2
        z_2 = x_1 + ((t_0 - 1) / t_1) * (x_1 - x_0)
        z_2 = z_2 + A.T @ (y - A @ z_2)

        # 软阈值处理
        x_2 = np.sign(z_2) * np.maximum(np.abs(z_2) - lambda_val, 0)

        # 计算误差
        error_1 = np.linalg.norm(x_2 - x_1) / np.linalg.norm(x_2)
        error_2 = np.linalg.norm(y - A @ x_2)
        error.append([error_1, error_2])

        # 判断是否满足误差阈值条件
        if error_1 < epsilon or error_2 < epsilon:
            break
        else:
            # 更新变量
            x_0 = x_1
            x_1 = x_2
            t_0 = t_1

    return x_2, np.array(error)


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
# 参数设定
N = 1024  # 信号长度
M = 512  # 测量数量
K = 10  # 稀疏信号中非零元素的数量

# 生成稀疏信号
x = np.zeros(N)
T = 5 * np.random.randn(K)  # 非零元素的值
index_k = np.random.permutation(N)  # 随机选择非零元素的位置
x[index_k[:K]] = T  # 将非零元素放置在信号中

# 生成测量矩阵
A = np.random.randn(M, N)
A = (1 / np.sqrt(M)) * A  # 归一化
A = orth(A.T).T  # 正交化

# 生成测量向量
y = A @ x  # + e (这里没有添加噪声)

# 调用FISTA算法恢复信号
x_rec1, error1 = cs_fista(y, A, lambda_val=5e-3, epsilon=1e-4, itermax=5000)
x_rec2, error2 = cs_ista(y, A, lambda_val=5e-3, epsilon=1e-4, itermax=5000)

# 绘制原始信号和恢复信号
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.stem(x, use_line_collection=True)
plt.title('Original Signal')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.subplot(3, 1, 2)
plt.stem(x_rec1, use_line_collection=True)
plt.title('Recovered Signal(fista)')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.subplot(3, 1, 3)
plt.stem(x_rec2, use_line_collection=True)
plt.title('Recovered Signal(ista)')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# 计算并输出均方根误差
rmse = np.sqrt(np.mean((x - x_rec1) ** 2))
rmse2 = np.sqrt(np.mean((x - x_rec2) ** 2))
print(f"原始信号与恢复信号之间的均方根误差 (RMSE): {rmse}")
print(f"原始信号与恢复信号之间的均方根误差 (RMSE): {rmse2}")