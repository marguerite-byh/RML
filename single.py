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
        if os.path.isfile(file_path) and file_name.endswith('.TXT'):
            specific_lines = read_specific_lines(file_path, lines_to_read)
            # 将每行数据分割并转换为数字（示例中假设每行数据以空格分隔）
            #matrix = [list(map(float, line.split())) for line in specific_lines]
            matrices.append(specific_lines)
    matrice = np.array(matrices)
    return matrice
# 文件夹路径
z=912 #频率信息
folder_path = (f'G:\\数据\\10.17-24\\9.30-22')#文件夹路径
result_matrices=[]
lines_to_read = [line for test in range(2) for line in range(test*2050+3, test*2050+2051)]# 要读取的行数列表前2048
# lines_to_read = [line for test in range(1,2) for line in range(test*2050+3, test*2050+2051)]# 要读取的行数列表后2048
# 读取并拼接矩阵
result_matrices=read_and_concat_matrices(folder_path,lines_to_read)
print(result_matrices.shape)
plt.imshow(result_matrices, cmap='jet', aspect='auto')
x_ticks = np.linspace(880, 960, 21) # 设置刻度的起始值、结束值和刻度数量
x_tick_labels = np.linspace(0, 4100, 21)  # 设置刻度对应的标签
plt.xticks(x_tick_labels, x_ticks)
plt.title(f"{z}")
plt.colorbar()  # 添加颜色条
plt.show()
