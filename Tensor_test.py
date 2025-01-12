
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import time
def generate_aggregated_signal_with_noise(N, M, K, cluster_size_range, snr):
    x_matrices = []
    centers = []
    center_values = []
    for _ in range(K):
        x = np.zeros((N, M))
        center_x = np.random.randint(0, N)
        center_y = np.random.randint(0, M)
        cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1])
        centers.append((center_x, center_y))
        sigma = cluster_size / 3
        for i in range(N):
            for j in range(M):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                if distance == 0:
                    x[i, j] += np.exp(-1 / (2 * sigma ** 2))
                else:
                    x[i, j] += np.exp(-distance ** 2 / (2 * sigma ** 2))

        x_matrices.append(x)
        center_values.append(np.mean(x ** 2))

    max_center_value = max(center_values)
    noise_power = max_center_value / (10 ** (snr / 10))
    aggregated_x = np.sum(x_matrices, axis=0)
    noise = np.abs(np.sqrt(noise_power) * np.random.randn(N, M))
    aggregated_x += noise

    return x_matrices, aggregated_x

# n, m = 100, 100
# N, M, K = 100, 100, 3
# cluster_size_range = (15, 25)
# original_matrix = generate_aggregated_signal(N, M, K, cluster_size_range)
N, M, K = 100, 100, 3
cluster_size_range = (5, 25)
SNR = 20  # dB
sample_rate = 0.15
x_matrices, aggregated_x = generate_aggregated_signal_with_noise(N, M, K, cluster_size_range, SNR)

aggregated_x = np.array(aggregated_x)
aggregated_x = ((aggregated_x - aggregated_x.min()) / (aggregated_x.max() - aggregated_x.min()) * 255).astype(np.uint8)
# 将 aggregated_x 扩展为三通道的图像
# 重复 aggregated_x 三次，并沿着新的轴（通道轴）堆叠
n = 5  # 你需要的层数
aggregated_x_stacked = np.stack([aggregated_x] * n, axis=-1)

# 保存为 .npy 文件
tensor_path = 'path_to_save_file.npy'
np.save(tensor_path, aggregated_x_stacked)

import cv2
import tensorly as tl
import numpy as np
import datetime
import matplotlib.pyplot as plt

def shrinkage(matrix, t):
    """矩阵的shrinkage运算
    Args:
        matrix: 进行shrinkage运算的矩阵
        t: 收缩算子
    Returns:
        shrinkageMatrix: 进行shrinkage运算以后的矩阵
    """
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
    sigm = np.zeros((U.shape[1], Vh.shape[0]))
    for i in range(len(S)):
        sigm[i, i] = np.max(S[i] - t, 0)
    temp = np.dot(U, sigm)
    shrinkageMatrix = np.dot(temp, Vh)
    return shrinkageMatrix


def readTensor(tensor_path):
    """读取三维张量数据
    Args:
        tensor_path: 张量数据路径

    Returns:
        tensor_data: 张量形式数据，形状为 [n, n, n]
    """
    # 假设 tensor_path 是一个保存 numpy 数组的文件路径
    tensor_data = np.load(tensor_path)
    # if tensor_data.ndim != 3:
    #     raise ValueError(f"张量的形状必须为 [n, n, n]，当前形状为: {tensor_data.shape}")
    return tensor_data


def missTensor(tensor_data, miss_percent):
    """对张量的数据进行部分缺失处理
    Args:
        tensor_data: 输入的三维张量数据，形状为 [n, n, n]
        miss_percent: 张量缺失数据百分比，取值为[0,1]

    Returns:
        sparse_tensor: 稀疏张量，只含有元素0和1，形状为 [n, n, n]
        miss_data_tensor: 缺失部分数据张量形式数据，形状为 [n, n, n]
    """
    imgSize = tensor_data.shape
    size = np.prod(imgSize)  # 张量总的元素数据
    missDataSize = int(np.ceil(np.prod([size, miss_percent])))  # 缺失元素数量
    nums = np.ones(size)  # 生成全为1的数组
    nums[:missDataSize] = 0  # 缺失的数据填充为0
    np.random.shuffle(nums)  # 对只含0,1的数组进行乱序排列
    sparse_tensor = tl.tensor(nums.reshape(imgSize))  # 生成只含有0,1的张量
    miss_data_tensor = sparse_tensor * tensor_data
    return sparse_tensor, miss_data_tensor


def createZeroTensor(tShape):
    return tl.zeros(tShape)


def HaLRTC(K, a, X, Z, rho, Y, origin_x):
    """HaLRTC算法实现
    Args:
        K: 最大迭代次数
        a: 核范数前的系数为一个数组
        X: 缺失部分数据的张量图片
        Z: 0-1张量
        rho: 罚参数
        Y: 初始时为0张量
        origin_x: 原始图片
    Returns:
        X_hat: 通过HaLRTC算法复原的图片张量形式数据
        rmse_list: 每次迭代的RMSE列表
    """
    Y1 = Y2 = Y3 = Y
    start_time = datetime.datetime.now()
    rmse_list = []  # 存储每次迭代的RMSE
    for k in range(K):
        i_start_time = datetime.datetime.now()
        print('iteration number is:{num}'.format(num=k + 1))
        # 1.更新Mi
        M1 = tl.fold(shrinkage(tl.unfold(X, mode=0) + tl.unfold(Y1, mode=0) / rho, a[0] / rho), 0, X.shape)
        M2 = tl.fold(shrinkage(tl.unfold(X, mode=1) + tl.unfold(Y2, mode=1) / rho, a[1] / rho), 1, X.shape)
        M3 = tl.fold(shrinkage(tl.unfold(X, mode=2) + tl.unfold(Y3, mode=2) / rho, a[2] / rho), 2, X.shape)
        # 2.更新X
        X_hat = (1 - Z) * (M1 + M2 + M3 - (Y1 + Y2 + Y3) / rho) / 3 + X * Z
        # 3.更新Lagrange乘子
        Y1 = Y1 - rho * (M1 - X_hat)
        Y2 = Y2 - rho * (M2 - X_hat)
        Y3 = Y3 - rho * (M3 - X_hat)

        rho = rho * 1.02
        i_end_time = datetime.datetime.now()
        cost = i_end_time - i_start_time
        print('the {num} times iteration ending,time cost is:{time} second'.format(num=k + 1, time=cost.seconds))

        error_norm = np.linalg.norm(origin_x.astype(np.float64) - X_hat.astype(np.float64))
        H_norm = np.linalg.norm(origin_x.astype(np.float64))
        rse_db = 10 * np.log10(error_norm ** 2 / H_norm ** 2)
        rmse_list.append(rse_db)

    end_time = datetime.datetime.now()
    print("total time cost: {} second".format((end_time - start_time).seconds))
    return X_hat, rmse_list


if __name__ == "__main__":
    np.random.seed(41)
    path = "path_to_save_file.npy"  # 假设这是一个保存 numpy 数组的文件路径
    # 原始张量
    origin_x = readTensor(path)
    # X为受损的张量
    Z, X = missTensor(origin_x, 0.7)
    imgShape = X.shape
    # 对应alpha
    a = abs(np.random.rand(3, 1))
    a = a / np.sum(a)
    K = 200  # 迭代次数
    rho = 1e-6
    Y = createZeroTensor(X.shape)
    # X_hat为通过算法还原的张量
    X_hat, rmse_list = HaLRTC(K, a, X, Z, rho, Y, origin_x)

    plt.figure(figsize=(15, 10))

    # 第一张图：原始张量
    plt.subplot(2, 2, 1)
    plt.imshow(origin_x[:, :, 0], cmap='jet')
    plt.title('Original Tensor')
    plt.axis('off')

    # 第二张图：缺失张量
    plt.subplot(2, 2, 2)
    plt.imshow(X[:, :, 0], cmap='jet')
    plt.title('Missing Tensor')
    plt.axis('off')

    # 第三张图：恢复后的张量
    plt.subplot(2, 2, 3)
    plt.imshow(X_hat[:, :, 0], cmap='jet')
    plt.title('Restored Tensor')
    plt.axis('off')

    # 第四张图：RMSE 曲线图
    plt.subplot(2, 2, 4)
    plt.plot(range(1, K + 1), rmse_list, marker='o')
    plt.xlabel('Iteration Number')
    plt.ylabel('RMSE (dB)')
    plt.title('RMSE vs Iteration Number')
    plt.grid(True)

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()

    # plt.close()  # 关闭当前图表