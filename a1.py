import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import numpy as np
# 或者使用 numpy 读取 CSV 文件
Signal_Sampled = np.loadtxt('Signal_Sampled.csv', delimiter=',')
Measure_Coords = np.loadtxt('Measure_Coords.csv', delimiter=',')
Signal_Noised_Vector = np.loadtxt('Signal_Noised_Vector.csv', delimiter=',')
Sample_Location = np.loadtxt('Sample_Location.csv', delimiter=',')
i=4
selected_data = Signal_Sampled[:, i]

first_column = Signal_Noised_Vector[:, i]
# 重塑为 51x51 的数组
reshaped_array = first_column.reshape(51, 51)

result_matrix = np.zeros((51, 51))
M,N=51,51

# 根据位置矩阵填充数据
Sample_Location-=1
Sample_Location = Sample_Location.astype(int)
result_matrix.flat[Sample_Location] = selected_data

data_sta = np.where(result_matrix == 0, np.nan, result_matrix)
lon_sta, lat_sta = np.meshgrid(np.arange(M), np.arange(N))
lon_sta, lat_sta = lon_sta.flatten().astype(float), lat_sta.flatten().astype(float)

# 获取有效的采样点
valid_idx = ~np.isnan(data_sta.flatten())
lon_valid = lon_sta[valid_idx]
lat_valid = lat_sta[valid_idx]
data_valid = data_sta.flatten()[valid_idx]


# 克里金插值
OK = OrdinaryKriging(
    lon_valid, lat_valid, data_valid,
    # variogram_model='hole-effect',
    variogram_model='spherical',
    verbose=False,
    enable_plotting=False,
    nlags=50,
    weight=True
)

lon2D, lat2D = np.meshgrid(np.arange(M).astype(float), np.arange(N).astype(float))
z, ss = OK.execute('grid', np.arange(M).astype(float), np.arange(N).astype(float))

# 绘制结果
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.title('Original Signal')
plt.imshow(reshaped_array, cmap='jet', origin='lower')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Observed Image")
plt.imshow(result_matrix, cmap='jet', origin='lower')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title('Recovered Signal')
plt.imshow(z, cmap='jet', origin='lower')
plt.colorbar()

plt.tight_layout()
plt.show()

try:
    rmse = np.sqrt(np.mean((z - reshaped_array) ** 2))
    relative_error = np.linalg.norm(z - reshaped_array) / np.linalg.norm(reshaped_array)
    plt.suptitle(f' Sample Rate=16%')
    # 打印结果
    plt.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nRelative Error: {relative_error:.4f}',
             transform=plt.gcf().transFigure,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5))

    print(f"Relative Error: {relative_error}")
    print(f"原始信号与恢复信号之间的均方根误差 (RMSE): {rmse}")
except Exception as e:
    print(f"计算过程中出现错误：{e}")

plt.show()
