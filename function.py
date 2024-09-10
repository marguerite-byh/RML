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
def delete_max_and_return_positions(matrix):
    positions = []
    new_matrix = []

    for i, row in enumerate(matrix):
        max_index = np.argmax(row)
        positions.append((i, max_index))
        new_row = np.delete(row, max_index)
        new_matrix.append(new_row)

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
def svt(M, P, T=None, delta=1, itermax=200, tol=1e-7):

    Y = np.zeros_like(M, dtype=float)
    iterations = 0

    if T is None:
        T = np.sqrt(M.shape[0] * M.shape[1])
    if delta is None:
        delta = 1
    if itermax is None:
        itermax = 200
    if tol is None:
        tol = 1e-7

    for ii in range(itermax):
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        S = np.sign(S) * np.maximum(np.abs(S) - T, 0)
        X = np.dot(U, np.dot(np.diag(S), Vt))
        Y = Y + delta * P * (M - X)
        Y = P * Y
        error = np.linalg.norm(P * (M - X), 'fro') / np.linalg.norm(P * M, 'fro')
        if error < tol:
            break
        iterations = ii + 1

    return X, iterations
def normalize(matrix):
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
def compute_rmse(X, M):
    X1=normalize(X)
    M1=normalize(M)
    return np.sqrt(np.mean((X1 - M1) ** 2))
def frobenius_norm_ratio(matrix1, matrix2):
    diff_matrix = matrix1 - matrix2
    norm_diff = np.linalg.norm(diff_matrix, 'fro')
    norm_matrix1 = np.linalg.norm(matrix1, 'fro')
    return norm_diff / norm_matrix1