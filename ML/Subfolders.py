#对文件夹里的所有子文件夹里的所有txt文件进行处理
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import glob
start_time = time.time()

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

def process_all_txt_files_in_subfolders(main_folder,lines_to_read):
    # 获取主文件夹中的所有子文件夹路径
    subfolders = [os.path.join(main_folder, subfolder) for subfolder in os.listdir(main_folder) if
                  os.path.isdir(os.path.join(main_folder, subfolder))]
    z = 1
    # 遍历每个子文件夹
    for subfolder in subfolders:
        matrices = []# 获取子文件夹中的所有TXT文件路径
        txt_files = glob.glob(os.path.join(subfolder, '*.txt'))
        # 遍历每个TXT文件并处理
        for txt_file in txt_files:
            specific_lines = read_specific_lines(txt_file, lines_to_read)
            matrices.append(specific_lines)
            matrice = np.array(matrices)
        plt.imshow(matrice, cmap='jet', aspect='auto')
        x_ticks = np.linspace(900, 940, 11)  # 设置刻度的起始值、结束值和刻度数量
        x_tick_labels = np.linspace(0, 4100, 11)  # 设置刻度对应的标签
        plt.xticks(x_tick_labels, x_ticks)
        plt.colorbar()  # 添加颜色条
        file_name = f"20_{subfolder}.png"
        plt.savefig(f"{main_folder}/{file_name}", bbox_inches='tight', pad_inches=0,dpi=400)  # 保存图像
        plt.close()
        print("第", z, "组图像保存完成！")
        z = z + 1

def read_and_concat_matrices(folder_path,group_files, lines_to_read):
    matrices = []
    for file_name in group_files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            specific_lines = read_specific_lines(file_path, lines_to_read)
            # 将每行数据分割并转换为数字（示例中假设每行数据以空格分隔）
            #matrix = [list(map(float, line.split())) for line in specific_lines]
            matrices.append(specific_lines)
            matrice = np.array(matrices)
    return matrice
# 文件夹路径

if __name__ == "__main__":
    main_folder = 'H:\\6.24早上'  # 替换为你的主文件夹路径
    lines_to_read = [line for test in range(2) for line in range(test * 2050 + 3, test * 2050 + 2051)]
    process_all_txt_files_in_subfolders(main_folder, lines_to_read)

