#绘制单个图
import os
import numpy as np
import matplotlib.pyplot as plt
import function

indices_to_delete = [1024, 2048,3072, 4096]
folder_path = f'H:\\6.26早上\\{5}'#文件夹路径
M=[]
lines_to_read = [line for test in range(2) for line in range(test*2050+3, test*2050+2051)]# 要读取的行数列表
# 读取并拼接矩阵
M=function.read_and_concat_matrices(folder_path,lines_to_read)
new_matrix, positions = function.delete_specific_indices(M, indices_to_delete)
n1, n2 = new_matrix.shape
sample_rate = 0.1
P = np.zeros(n1 * n2)  # 初始化为一维数组
MM = new_matrix.flatten()
pos = np.sort(np.random.choice(n1 * n2, int(n1 * n2 * sample_rate), replace=False))
P[pos] = MM[pos]  # 将观测值赋给 P
index1 = np.where(P)[0]
P[index1] = 1
P = P.reshape(n1, n2)  # 重塑为矩阵
print(P)
# 设置阈值和步长
T = np.sqrt(n1 * n2)
delta = 0.25
# 使用 SVT 算法进行矩阵补全
X, iterations = function.svt(new_matrix, P, T, delta)
# 计算输出均方根误差
rmse = function.compute_rmse(X, new_matrix)
print("均方根误差: ", rmse)
#print(read_specific_lines('H:\\test3\\test4\\data001.txt', lines_to_read))
# print(result_matrices)
new_matrix, positions = function.delete_specific_indices(M, indices_to_delete)

print(new_matrix.shape)

plt.subplot(1, 3, 1)
plt.imshow(new_matrix, cmap='jet', aspect='auto')
x_ticks = np.linspace(900, 940, 11)
x_tick_labels = np.linspace(0, 4100, 11)
plt.xticks(x_tick_labels, x_ticks)
plt.colorbar()
plt.title("True Image")

plt.subplot(1, 3, 2)
plt.imshow(P*new_matrix, cmap='jet', aspect='auto')
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
plt.title("Restructed Image")
plt.show()