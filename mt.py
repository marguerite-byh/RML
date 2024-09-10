#绘制单个图
import os
import numpy as np
import matplotlib.pyplot as plt
def read_specific_lines(file_path, lines_to_read):
    specific_lines = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            if i in lines_to_read:
                specific_lines.append(line.strip())    # 去除换行符并添加到列表中
    # # 使用 map() 函数去除外层的列表
    # specific_lines = list(map(lambda x: x[0], specific_lines))
    specific_lines = [int(item) for item in specific_lines]
    return specific_lines
def read_and_concat_matrices(folder_path, lines_to_read):
    matrices = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            specific_lines = read_specific_lines(file_path, lines_to_read)
            # 将每行数据分割并转换为数字（示例中假设每行数据以空格分隔）
            #matrix = [list(map(float, line.split())) for line in specific_lines]
            matrices.append(specific_lines)
    matrice = np.array(matrices)
    return matrice
# 文件夹路径
z=1
folder_path1 = f'H:\\antenna\\1'#文件夹路径
folder_path2 = f'H:\\antenna\\2.1'#文件夹路径
folder_path3 = f'H:\\antenna\\3.1'#文件夹路径
result_matrices1=[]
result_matrices2=[]
result_matrices3=[]
lines_to_read = [line for test in range(2) for line in range(test*2050+3, test*2050+2051)]# 要读取的行数列表
# 读取并拼接矩阵
result_matrices1=read_and_concat_matrices(folder_path1,lines_to_read)
result_matrices2=read_and_concat_matrices(folder_path2,lines_to_read)
result_matrices3=read_and_concat_matrices(folder_path3,lines_to_read)
#print(read_specific_lines('H:\\test3\\test4\\data001.txt', lines_to_read))
# print(result_matrices)
matrices = [result_matrices1, result_matrices2, result_matrices3]
# 找到所有矩阵中的最小值和最大值
min_val = min(matrix.min() for matrix in matrices)
max_val = max(matrix.max() for matrix in matrices)
# 归一化函数
def normalize(matrix, min_val, max_val):
    return (matrix - min_val) / (max_val - min_val)
# 归一化所有矩阵
normalized_matrices = [normalize(matrix, min_val, max_val) for matrix in matrices]

# 创建子图
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
x_ticks = np.linspace(900, 940, 11)
x_tick_labels = np.linspace(0, 4100, 11)
for ax, matrix, i in zip(axs, normalized_matrices, range(1, 4)):
    cax = ax.imshow(matrix, cmap='jet', aspect='auto')
    ax.set_title(f'Matrix {i}')
    # 设置自定义的x轴刻度和标签
    ax.set_xticks(x_tick_labels)
    ax.set_xticklabels([f'{int(x)}' for x in x_ticks])
    fig.colorbar(cax, ax=ax)

plt.tight_layout()
plt.show()
