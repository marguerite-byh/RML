import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# 读取图像
A = plt.imread('test.jpg')

# 将图像数据转换为double类型
WW = A.astype(np.float64)
a1 = WW[:, :, 0]
a2 = WW[:, :, 1]
a3 = WW[:, :, 2]
M, N, _ = WW.shape
X = np.zeros((M, N, 3), dtype=np.float64)

# 对每个颜色通道进行处理
for jj, channel in enumerate([a1, a2, a3]):
    lambda_param = 1 / np.sqrt(max(M, N))
    u = 1 * lambda_param
    pca = PCA(n_components=1, svd_solver='auto', tol=1e-8)
    transformed = pca.fit_transform(channel.reshape(-1, 1))
    X[:, :, jj] = pca.inverse_transform(transformed).reshape(M, N)
    S = channel - X[:, :, jj]

# 显示原图
plt.figure(1)
plt.subplot(3, 1, 1)
plt.imshow(A.astype(np.uint8))
plt.title("原图", fontsize=12)

# 显示低秩图
plt.subplot(3, 1, 2)
plt.imshow(X.astype(np.uint8))
plt.title("低秩图", fontsize=12)

# 显示噪声图
d = S.copy()
d[d < 20] = 255
plt.subplot(3, 1, 3)
plt.imshow(d.astype(np.uint8))
plt.title("噪声图", fontsize=12)

plt.show()
