import os
import time
import numpy as np
import matplotlib.pyplot as plt
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
folder_path = 'H:\\gsm'
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
group_size = 20
z=1# 要读取的行数列表
for j in range(0, len(txt_files), group_size):
    if j + group_size <= len(txt_files):
        group_files = txt_files[j:j + group_size]
        result_matrices=[]
        for i in range(0,5,1):
            start=i*2050+3
            lines_to_read = list(range(start,start+2048,1))
            result_matrices.append(read_and_concat_matrices(folder_path, group_files,lines_to_read))

        print(result_matrices)
#         plt.imshow(result_matrices, cmap='jet', aspect='auto')
#         x_ticks = np.linspace(900, 1000, 20)  # 设置刻度的起始值、结束值和刻度数量
#         x_tick_labels = np.linspace(0, 10240, 5)  # 设置刻度对应的标签
#         plt.xticks(x_tick_labels, x_ticks)
#         plt.colorbar()  # 添加颜色条
#         # plt.axis('off')
#         file_name = f"gsm_image1_{z}.png"
#         plt.savefig(f"H:/gsm1/{file_name}", bbox_inches='tight', pad_inches=0)  # 保存图像
#         plt.close()
#
#     print("第",z,"组图像保存完成！")
#     z = z + 1
# end_time = time.time()
# # 计算执行时间
# execution_time = end_time - start_time
# # 输出执行时间
# print("程序执行时间：", execution_time, "秒")
#
# print("图像保存完成！")



# for i, data in enumerate(data_list):
#     # 绘制灰度图
#     plt.imshow(data, cmap='gray')
#     plt.axis('off')  # 关闭坐标轴
#     file_name = f"gray_image_{i}.png"  # 根据循环索引生成文件名
#     plt.savefig(f"H:/pic/{file_name}", bbox_inches='tight', pad_inches=0)  # 保存图像
#     plt.close()  # 关闭图像窗口，防止内存溢出
#
# print("图像保存完成！")
