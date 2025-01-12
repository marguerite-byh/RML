import os

def read_specific_line(file_path, line_number):
    """
    读取文件的指定行数据。

    :param file_path: 文件路径
    :param line_number: 要读取的行号
    :return: 指定行的数据，如果文件不足指定行，则返回 None
    """
    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            if i == line_number:
                return line.strip()
    return None

def read_specific_lines_from_folder(folder_path, line_number):
    """
    从指定文件夹中的所有 .txt 文件中读取指定行数据，并拼接成一个列表。

    :param folder_path: 文件夹路径
    :param line_number: 要读取的行号
    :return: 包含所有文件指定行数据的列表
    """
    data_list = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.TXT'):
            line = read_specific_line(file_path, line_number)
            if line is not None:
                data_list.append(line)
    return data_list

# 文件夹路径
folder_path = f'G:\\数据\\11.1-24\\21。30-9.30'  # 替换为你的文件夹路径

# 指定要读取的行号
line_number = 727 # 可以改为任意行号

# 读取指定行的数据并拼接成列表
data_list = read_specific_lines_from_folder(folder_path, line_number)

# 打印结果
print(data_list)
