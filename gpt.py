from matplotlib import pyplot as plt
from scipy import stats
import function
import numpy as np
indices_to_delete = [1024, 2048,3072, 4096]
folder_path = f'H:\\tst\\t'#文件夹路径
M=[]
lines_to_read = [line for test in range(2) for line in range(test*2050+3, test*2050+2051)]# 要读取的行数列表
# 读取并拼接矩阵
M=function.read_and_concat_matrices(folder_path,lines_to_read)
def delete_top_n_max_and_return_positions(matrix, n):
    positions = []
    new_matrix = []

    for i in range(matrix.shape[0]):
        row_positions = []
        row = list(matrix[i])
        for _ in range(n):
            max_index = np.argmax(row)
            row_positions.append(max_index)
            row[max_index] = -np.inf  # 将最大值设置为无穷小以找到下一个最大值
        row_positions.sort(reverse=True)
        for index in row_positions:
            row.pop(index)
        positions.append(row_positions)
        new_matrix.append(row)

    return np.array(new_matrix), positions


def calculate_mode_and_median(matrix):
    modes = []
    medians = []

    for row in matrix:
        mode = stats.mode(row)[0][0]
        median = np.median(row)
        modes.append(mode)
        medians.append(median)

    return modes, medians
new_matrix, positions = delete_top_n_max_and_return_positions(M, 9)
plt.imshow(new_matrix, cmap='jet', aspect='auto')
x_ticks = np.linspace(900, 940, 11)  # 设置刻度的起始值、结束值和刻度数量
x_tick_labels = np.linspace(0, 4100, 11)  # 设置刻度对应的标签
plt.xticks(x_tick_labels, x_ticks)
plt.colorbar()  # 添加颜色条