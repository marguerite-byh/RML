import numpy as np
# 初始化变量
from single import read_and_concat_matrices,read_specific_lines

# 假设 pr_power_sort_100 是一个包含元素的列表
folder_path = 'H:\\test3\\test4'
# 要读取的行数列表
lines_to_read = list(range(4,2052,1))
# 读取并拼接矩阵
result_matrices = read_and_concat_matrices(folder_path, lines_to_read)
# 对列表进行排序，但不改变原列表
for slist in result_matrices:
    index = 0
    count = []
    pr_power_100 = slist
    sorted_list = sorted(pr_power_100, reverse=True)
# 计算前一半元素的和
    half_length = len(sorted_list) // 2
    sum_power = sum(sorted_list[:half_length])
    sum_average_power = sum_power / (half_length )
    print("sub sum_average_power is", sum_average_power)

    while True:
        index += 1
        b_test = 0.0
        lenth = 0
        # print(index)
        # 计算小于阈值的元素个数并累加和
        for pr_power_100_value in pr_power_100:
            if pr_power_100_value < sum_average_power:
                lenth += 1
                b_test += pr_power_100_value

        count.append(lenth)

        # 判断终止条件
        if index > 1 and count[index - 1] == count[index - 2]:
            sum_average_power = 2.5*(b_test / lenth)
            # print("sum_average_power is", sum_average_power)
            break

        sum_average_power = 2.5 * (b_test / lenth)
    print("sum_average_power is", sum_average_power)
    new_list = [1 if x > sum_average_power else 0 for x in slist]
    print(sum(new_list))

