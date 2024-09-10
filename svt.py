import numpy as np
import matplotlib.pyplot as plt
import cv2
#SVT的图片实现
def svt(M, P, T=None, delta=1, itermax=200, tol=1e-7):
    """
    Singular Value Thresholding (SVT) algorithm
    Function to solve the following optimization problem:
                      min  ||X||_*
                   s.t. P(X - M) = 0
    X: recovered matrix
    M: observed matrix
    P: mask matrix, 1 for observed entries, 0 for missing entries
    T: singular value threshold
    delta: step size
    itermax: maximum number of iterations
    tol: tolerance for convergence
    Returns:
    X: recovered matrix
    iterations: number of iterations performed
    """
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
def compute_rmse(X, M):
    """
    Compute the Root Mean Square Error (RMSE) between the recovered matrix X and the original matrix M
    """
    return np.sqrt(np.mean((X - M) ** 2))
# Read image
image_path = r'H:/1/1.png'
A = cv2.imread(image_path)

if A is None:
    print(f"Error: Unable to read image at {image_path}")
else:
    print(f"Image read successfully from {image_path}")

# Ensure image is not empty
if A is not None:
    A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
    WW = A.astype(float)
    a1 = WW[:, :, 0]
    a2 = WW[:, :, 1]
    a3 = WW[:, :, 2]

    M, N = a1.shape
    X = np.zeros((M, N, 3))
    # Sampling
    pos = np.sort(np.random.permutation(M * N)[:int(M * N * 0.5)])

    for jj in range(3):
        P = np.zeros(M * N)
        MM = WW[:, :, jj].flatten()
        P[pos] = MM[pos]
        index1 = np.nonzero(P)[0]
        P[index1] = 1
        P = P.reshape(M, N)
        T = 50000
        delta_t = 1
        X[:, :, jj], iterations = svt(WW[:, :, jj], P, T, delta_t)

    DD = P * WW[:, :, 0]
    DD1 = P * WW[:, :, 1]
    DD2 = P * WW[:, :, 2]
    ff = np.zeros_like(WW)
    ff[:, :, 0] = DD
    ff[:, :, 1] = DD1
    ff[:, :, 2] = DD2
    rmse = compute_rmse(X, A)
    print("均方根误差: ", rmse)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(A.astype(np.uint8),cmap='jet', aspect='auto')
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(ff.astype(np.uint8),cmap='jet', aspect='auto')
    plt.title("Original Image")

    plt.subplot(1, 3, 3)
    plt.imshow(X.astype(np.uint8),cmap='jet', aspect='auto')
    plt.title("Recovered Image")

    plt.show()
