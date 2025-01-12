import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

def read_specific_lines(file_path, lines_to_read):
    def read_lines():
        try:
            with open(file_path, 'r') as file:
                for i, line in enumerate(file, 1):
                    if i in lines_to_read:
                        yield line.strip()
        except FileNotFoundError:
            print(f"文件 {file_path} 不存在")
        except IOError:
            print(f"无法读取文件 {file_path}")

    specific_lines = [int(item) for item in read_lines()]
    return specific_lines

def read_and_concat_matrices(folder_path, lines_to_read):
    matrices = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.TXT'):
            specific_lines = read_specific_lines(file_path, lines_to_read)
            matrices.append(np.array(specific_lines))

    if matrices:
        matrice = np.vstack(matrices)
    else:
        matrice = np.array([])
    return matrice

# 文件夹路径
z = 912  # 频率信息
folder_path = (f'G:\\数据\\10.17-24\\9.30-22')  # 文件夹路径
result_matrices = []
lines_to_read = list(chain.from_iterable(range(test * 2050 + 3, test * 2050 + 2051) for test in range(2)))  # 要读取的行数列表前2048

# 读取并拼接矩阵
result_matrices = read_and_concat_matrices(folder_path, lines_to_read)
print(result_matrices.shape)

try:
    plt.imshow(result_matrices, cmap='jet', aspect='auto')
except ValueError as e:
    print(f"图像显示错误: {e}")

x_ticks = np.linspace(880, 960, 21)  # 设置刻度的起始值、结束值和刻度数量
x_tick_labels = np.linspace(0, 4100, 21)  # 设置刻度对应的标签
plt.xticks(x_tick_labels, x_ticks)
plt.title(f"{z}")
plt.colorbar()  # 添加颜色条
plt.show()
