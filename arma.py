import cvxpy
import numpy as np
from scipy.spatial.distance import cdist



# === 1. 生成模拟数据 ===
def generate_tensor(shape, missing_rate=0.3):
    """生成一个低秩张量并添加缺失值"""
    np.random.seed(42)
    A = np.random.rand(shape[0], 5)
    B = np.random.rand(shape[1], 5)
    C = np.random.rand(shape[2], 5)
    tensor = np.einsum('ik,jk,mk->ijm', A, B, C)  # 构造低秩张量

    # 添加缺失值
    mask = np.random.rand(*shape) > missing_rate  # 保留的值
    tensor_with_nan = tensor * mask
    tensor_with_nan[mask == 0] = np.nan
    return tensor, tensor_with_nan, mask


# 模拟数据
true_tensor, observed_tensor, mask = generate_tensor((20, 20, 10), missing_rate=0.4)


# === 2. 反距离加权插值 (IDW) ===
def idw_interpolation(tensor):
    """使用反距离加权插值填补缺失值"""
    filled_tensor = tensor.copy()
    for i in range(tensor.shape[2]):  # 对每个变量维度单独处理
        layer = tensor[:, :, i]
        missing_indices = np.argwhere(np.isnan(layer))
        known_indices = np.argwhere(~np.isnan(layer))
        known_values = layer[~np.isnan(layer)]

        # 计算距离矩阵
        distances = cdist(missing_indices, known_indices)
        distances[distances == 0] = 1e-6  # 避免除以零
        weights = 1 / distances ** 2  # 距离反比权重

        # 插值
        interpolated_values = np.sum(weights * known_values, axis=1) / np.sum(weights, axis=1)
        for idx, val in zip(missing_indices, interpolated_values):
            filled_tensor[idx[0], idx[1], i] = val
    return filled_tensor


# 使用 IDW 填充初始缺失值
idw_tensor = idw_interpolation(observed_tensor)


# === 3. 核范数最小化张量补全 ===


def tensor_completion(idw_tensor, mask, lambda_reg=1):
    # 假设 idw_tensor 和 mask 都是三维张量
    observed_indices = np.where(mask)

    # 将张量展平为矩阵
    num_rows = idw_tensor.shape[0] * idw_tensor.shape[1]
    num_cols = idw_tensor.shape[2]
    X = cvxpy.Variable((num_rows, num_cols), name="X")  # 这里添加了变量名"X"

    # 展平 idw_tensor 和 mask
    flat_idw_tensor = idw_tensor.reshape(-1, idw_tensor.shape[2])
    flat_mask = mask.reshape(-1, idw_tensor.shape[2])

    # 计算目标函数
    objective = cvxpy.Minimize(
        cvxpy.norm(X, 'nuc') + lambda_reg * cvxpy.norm(X[observed_indices[0], :] - flat_idw_tensor[observed_indices[0], :], 'fro'))

    # 定义约束条件
    constraints = [X >= 0]

    # 创建并求解问题
    problem = cvxpy.Problem(objective, constraints)
    problem.solve()

    # 将结果重新 reshape 为原始张量形状
    completed_tensor = X.value.reshape(idw_tensor.shape)

    return completed_tensor


# 使用核范数最小化优化结果
completed_tensor = tensor_completion(idw_tensor, mask)

# === 4. 可视化结果 ===
import matplotlib.pyplot as plt


def plot_tensor_slices(true_tensor, completed_tensor, slice_idx=5):
    """对比真实张量和补全张量的某个切片"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(true_tensor[:, :, slice_idx], cmap='viridis')
    axes[0].set_title("True Tensor (Slice {})".format(slice_idx))
    axes[1].imshow(completed_tensor[:, :, slice_idx], cmap='viridis')
    axes[1].set_title("Completed Tensor (Slice {})".format(slice_idx))
    plt.show()


# 显示结果
plot_tensor_slices(true_tensor, completed_tensor)
