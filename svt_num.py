import numpy as np
import matplotlib.pyplot as plt
from single import read_specific_lines, read_and_concat_matrices
import function as f2
def svt(M, P, T=None, delta=1, itermax=200, tol=1e-7):

    Y = np.zeros_like(M, dtype=float)
    iterations = 0

    if T is None:
        T = np.sqrt(M.shape[0] * M.shape[1])
    if delta is None:
        delta = 1
    if itermax is None:
        itermax = 200
    if tol is None:
        tol = 1e-7

    for ii in range(itermax):
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        S = np.sign(S) * np.maximum(np.abs(S) - T, 0)
        X = np.dot(U, np.dot(np.diag(S), Vt))
        Y = Y + delta * P * (M - X)
        Y = P * Y
        error = np.linalg.norm(P * (M - X), 'fro') / np.linalg.norm(P * M, 'fro')
        if error < tol:
            break
        iterations = ii + 1

    return X, iterations
def normalize(matrix):
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
def compute_rmse(X, M):
    X1=normalize(X)
    M1=normalize(M)
    return np.sqrt(np.mean((X1 - M1) ** 2))
def frobenius_norm_ratio(matrix1, matrix2):
    diff_matrix = matrix1 - matrix2
    norm_diff = np.linalg.norm(diff_matrix, 'fro')
    norm_matrix1 = np.linalg.norm(matrix1, 'fro')
    return norm_diff / norm_matrix1
# 数值验证
folder_path = f'H:\\1\\{1}'#文件夹路径
M=[]
lines_to_read = [line for test in range(2) for line in range(test*2050+3, test*2050+2051)]# 要读取的行数列表

# 读取并拼接矩阵
M=read_and_concat_matrices(folder_path,lines_to_read)


n1, n2 = M.shape

# 采样率
sample_rate = 0.1

# 构造采样矩阵
P = np.zeros(n1 * n2)  # 初始化为一维数组
MM = M.flatten()
pos = np.sort(np.random.choice(n1 * n2, int(n1 * n2 * sample_rate), replace=False))
P[pos] = MM[pos]  # 将观测值赋给 P
index1 = np.where(P)[0]
P[index1] = 1
P = P.reshape(n1, n2)  # 重塑为矩阵
print(P)
# 设置阈值和步长
T = np.sqrt(n1 * n2)
delta = 2

# 使用 SVT 算法进行矩阵补全
X, iterations = svt(M, P, T, delta)

# 计算并输出均方根误差
rmse = compute_rmse(X, M)
print("均方根误差: ", rmse)

X=f2.normalize(X)
M=f2.normalize(M)
f2=frobenius_norm_ratio(X, M)
print("frobenius_norm_ratio: ", f2)
# 可视化结果
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
plt.imshow(X, cmap='jet', aspect='auto')
x_ticks = np.linspace(900, 940, 11)
x_tick_labels = np.linspace(0, 4100, 11)
plt.xticks(x_tick_labels, x_ticks)
plt.colorbar()
plt.title("Recovered Image")

plt.show()