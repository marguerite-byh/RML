import numpy as np
import matplotlib.pyplot as plt
def cs_fista(y, A, lambda_=2e-5, epsilon=1e-4, itermax=10000):
    N = A.shape[1]
    error = []

    x_0 = np.zeros(N)
    x_1 = np.zeros(N)
    t_0 = 1

    for i in range(itermax):
        t_1 = (1 + np.sqrt(1 + 4 * t_0 ** 2)) / 2
        alpha = 1
        z_2 = x_1 + ((t_0 - 1) / t_1) * (x_1 - x_0)

        # 加入数值稳定性检查
        z_2 = np.nan_to_num(z_2)  # 将无效值替换为0
        try:
            A_z_2 = A @ z_2
            y_minus_A_z_2 = y - A_z_2
            A_T_y_minus_A_z_2 = A.T @ y_minus_A_z_2
        except FloatingPointError:
            continue

        z_2 = z_2 + A_T_y_minus_A_z_2
        x_2 = np.sign(z_2) * np.maximum(np.abs(z_2) - alpha * lambda_, 0)

        # 计算重建误差
        norm_x2 = np.linalg.norm(x_2)
        rel_error = np.linalg.norm(x_2 - x_1) / norm_x2 if norm_x2 != 0 else 0
        meas_error = np.linalg.norm(y - A @ x_2)

        error.append((rel_error, meas_error))

        if rel_error < epsilon or meas_error < epsilon:
            break
        else:
            x_0 = x_1
            x_1 = x_2
            t_0 = t_1

    return x_2, np.array(error)
# 数据标准化函数
def normalize_data(A, y):
    A_norm = A / np.linalg.norm(A, axis=0)
    y_norm = y / np.linalg.norm(y)
    return A_norm, y_norm
# 示例数据生成与验证
np.random.seed(0)
m, n = 50, 100
# 生成稀疏信
x_true = np.zeros(n)
x_true[np.random.choice(n, 10, replace=False)] = np.random.randn(10)

# 生成测量矩阵和测量向量
A = np.random.randn(m, n)
y = A @ x_true + 0.01 * np.random.randn(m)  # 添加噪声

# 标准化数据
A_norm, y_norm = normalize_data(A, y)

# 缩小矩阵 A 的值范围
A_norm = A_norm / np.sqrt(m)

# 使用FISTA算法恢复原始信号
x_hat, error = cs_fista(y_norm, A_norm)

# 绘制恢复信号、原始信号和误差曲线
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 原始信号
axes[0].stem(x_true, use_line_collection=True)
axes[0].set_title('Original Signal (x_true)')
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Amplitude')

# 恢复信号
axes[1].stem(x_hat, use_line_collection=True)
axes[1].set_title('Recovered Signal (x_hat)')
axes[1].set_xlabel('Index')
axes[1].set_ylabel('Amplitude')

# 误差曲线
axes[2].plot(error[:, 0], label='Relative Error')
axes[2].plot(error[:, 1], label='Measurement Error')
axes[2].set_title('Error Curves')
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('Error')
axes[2].legend()

plt.tight_layout()
plt.show()

print("Recovered signal (x_hat):", x_hat)
print("Original signal (x_true):", x_true)
print("Relative Error:", np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true))
print("Measurement Error:", np.linalg.norm(y - A @ x_hat))
