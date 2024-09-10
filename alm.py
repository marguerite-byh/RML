import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import function as f2

def sig_thre(M, tau):
    U, Sigma, Vt = svd(M, full_matrices=False)
    Sigma = np.maximum(Sigma - tau, 0)
    return U @ np.diag(Sigma) @ Vt


def soft_thre(M, tau):
    return np.sign(M) * np.maximum(np.abs(M) - tau, 0)


def pcp_ad(M, u=None, lambda_=None, itemax=1000, tol=1e-6):
    m, n = M.shape
    S = np.zeros((m, n))
    Y = np.zeros((m, n))
    L = np.zeros((m, n))

    M = np.nan_to_num(M)

    if lambda_ is None:
        lambda_ = 1 / np.sqrt(max(m, n))
    if u is None:
        u = 10 * lambda_
    if itemax is None:
        itemax = 1000
    if tol is None:
        tol = 1e-6

    for ii in range(itemax):
        L = sig_thre(M - S + (1 / u) * Y, 1 / u)
        S = soft_thre(M - L + (1 / u) * Y, lambda_ / u)
        Z = M - L - S
        Y = Y + u * Z
        error = np.linalg.norm(M - L - S, 'fro') / np.linalg.norm(M, 'fro')
        if (ii == 0) or (ii % 10 == 0) or (error < tol):
            print(
                f'iter: {ii:04d}\terr: {error:.6f}\trank(L): {np.linalg.matrix_rank(L)}\tcard(S): {np.count_nonzero(S)}')
        if error < tol:
            break

    return L, S


# 示例数据生成与验证
# np.random.seed(0)
# m, n = 50, 100
#
# # 生成低秩矩阵 L_true
# L_true = np.random.randn(m, 10) @ np.random.randn(10, n)
#
# # 生成稀疏矩阵 S_true
# S_true = np.random.randn(m, n)
# S_true[np.abs(S_true) < 2.5] = 0  # 设置为稀疏

# # 原始矩阵 X_true
# X_true = L_true + S_true
folder_path = f'H:\\1\\{1}'#文件夹路径
M=[]
lines_to_read = [line for test in range(2) for line in range(test*2050+3, test*2050+2051)]# 要读取的行数列表

# 读取并拼接矩阵
M=f2.read_and_concat_matrices(folder_path,lines_to_read)
# # 随机采样，得到采样矩阵 M_sampled
# mask = np.random.rand(m, n) < 0.5  # 50%的采样率
# M_sampled = X_true.copy()
# M_sampled[~mask] = np.nan
n1, n2 = M.shape
# 采样率
sample_rate = 0.9
# 构造采样矩阵
P = np.zeros(n1 * n2)  # 初始化为一维数组
MM = M.flatten()
pos = np.sort(np.random.choice(n1 * n2, int(n1 * n2 * sample_rate), replace=False))
P[pos] = MM[pos]  # 将观测值赋给 P
index1 = np.where(P)[0]
P[index1] = 1
P = P.reshape(n1, n2)  # 重塑为矩阵
print(P)
# 使用PCP算法恢复原始矩阵
L, S = pcp_ad(M*P)
t=L+S
t=f2.normalize(t)
M=f2.normalize(M)
f=f2.frobenius_norm_ratio(t, M)
print("frobenius_norm_ratio: ", f)
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(M, cmap='jet', aspect='auto')
x_ticks = np.linspace(900, 940, 11)
x_tick_labels = np.linspace(0, 4100, 11)
plt.xticks(x_tick_labels, x_ticks)
plt.colorbar()
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(P * M, cmap='jet', aspect='auto')
x_ticks = np.linspace(900, 940, 11)
x_tick_labels = np.linspace(0, 4100, 11)
plt.xticks(x_tick_labels, x_ticks)
plt.colorbar()
plt.title("Oberseved Image")

plt.subplot(1, 3, 3)
plt.imshow(t, cmap='jet', aspect='auto')
x_ticks = np.linspace(900, 940, 11)
x_tick_labels = np.linspace(0, 4100, 11)
plt.xticks(x_tick_labels, x_ticks)
plt.colorbar()
plt.title("Recovered Image")

plt.show()
