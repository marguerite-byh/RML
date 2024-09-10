import os
import time
import numpy as np
import matplotlib.pyplot as plt
start_time = time.time()
#文件下所有txt文件，20个一组绘制图像
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

def delete_specific_indices(matrix, indices):
    positions = []
    new_matrix = []

    for row in matrix:
        row_positions = []
        new_row = list(row)
        for index in sorted(indices, reverse=True):
            if index < len(new_row):
                row_positions.append(index)
                new_row.pop(index)
        positions.append(row_positions)
        new_matrix.append(new_row)

    return np.array(new_matrix), positions
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
folder_path = 'H:\\a'
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
indices_to_delete = [1024,2048,3072, 4096,1023,1025,2047,2049,3071,3073,4095,4097,4094,4098,1,0]
group_size = 20
z=1# 要读取的行数列表
for j in range(0, len(txt_files), group_size):
    if j + group_size <= len(txt_files):
        group_files = txt_files[j:j + group_size]
        lines_to_read = [line for test in range(10) for line in range(test*2050+3, test*2050+2051)]
        result_matrices = read_and_concat_matrices(folder_path, group_files,lines_to_read)
        new_matrix, positions = delete_specific_indices(result_matrices, indices_to_delete)
        plt.imshow(result_matrices, cmap='jet', aspect='auto')
        x_ticks = np.linspace(940, 980, 11)  # 设置刻度的起始值、结束值和刻度数量
        x_tick_labels = np.linspace(0, 4100, 11)  # 设置刻度对应的标签
        plt.xticks(x_tick_labels, x_ticks)
        plt.colorbar()  # 添加颜色条
        # plt.axis('off')
        file_name = f"20_{z}.png"
        plt.savefig(f"H:/a/{file_name}", bbox_inches='tight', pad_inches=0)  # 保存图像
        plt.close()

    print("第",z,"组图像保存完成！")
    z = z + 1
end_time = time.time()
# 计算执行时间
execution_time = end_time - start_time
# 输出执行时间
print("程序执行时间：", execution_time, "秒")

print("图像保存完成！")

