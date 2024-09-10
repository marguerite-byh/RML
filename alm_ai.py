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


# 参数设定
N = 100 # 信号的行数
M = 100  # 信号的列数
P = 765  # 测量数量
K = 100  # 稀疏信号中非零元素的数量

# 生成二维稀疏信号
x = np.zeros((N, M))
T = 5 * np.random.randn(K)  # 非零元素的值
index_k = np.random.permutation(N * M)  # 随机选择非零元素的位置
x_flat = x.flatten()
x_flat[index_k[:K]] = T  # 将非零元素放置在信号中
x = x_flat.reshape(N, M)

# 生成测量矩阵
A = np.random.randn(P, N * M)
A = (1 / np.sqrt(P)) * A  # 归一化
A = orth(A.T).T  # 正交化

# 生成测量向量
y = A @ x.flatten()  # + e (这里没有添加噪声)

# 调用ISTA算法恢复信号
x_rec_flat, error1 = cs_ista(y, A, lambda_val=5e-3, epsilon=1e-4, itermax=5000)
x_rec = x_rec_flat.reshape(N, M)

# 绘制原始信号和恢复信号
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.imshow(x, cmap='gray')
plt.title('原始信号')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.imshow(x_rec, cmap='gray')
plt.title('恢复信号')
plt.colorbar()

plt.tight_layout()
plt.show()

# 计算并输出均方根误差
rmse = np.linalg.norm(x_rec-x, 'fro') / np.linalg.norm(x, 'fro')
print(f"原始信号与恢复信号之间的均方根误差 (RMSE): {rmse}")
