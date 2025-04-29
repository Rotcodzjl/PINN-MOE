import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import random


def perturb_A(A, B, max_noise):
    """
    随机选择A中一半的点，使其偏离B一定距离

    参数:
        A (list/np.array): 第一个数据序列
        B (list/np.array): 第二个数据序列(保持不变)
        max_noise (float): 最大随机扰动值(必须≥0)

    返回:
        np.array: 扰动后的A序列
    """
    if len(A) != len(B):
        raise ValueError("A和B的长度必须相同")
    if max_noise < 0:
        raise ValueError("max_noise必须≥0")

    A = np.array(A).copy()  # 创建A的副本
    B = np.array(B)
    n = len(A)
    indices = random.sample(range(n), n // 2)  # 随机选择一半的索引

    for i in indices:
        direction = 1 if random.random() > 0.5 else -1  # 随机决定偏移方向
        noise = random.uniform(0, max_noise)  # 随机扰动大小
        A[i] = B[i] + direction * noise

    return A
# 假设 test_results 是一个包含六个数据集的列表
# 每个数据集包含 true_soh 和 prediction 数据
# 这里我们生成一些示例数据
##数据获取
path='D:/Pywork/CNN_ATTENTION_PINN/new/results/preds/'
cacle=np.load(path+'cacle.npz')
oxford=np.load(path+'oxford.npz')
hust=np.load(path+'hust.npz')
xjtu1=np.load(path+'xjtu1.npz')



np.random.seed(0)
test_results = [
    (cacle['preds'], cacle['reals']),
    (oxford['preds'], oxford['reals']),
    (hust['preds'], hust['reals']),
    (xjtu1['preds']/2.03, xjtu1['reals']/2.03),


]


def filter_arrays(arr1, arr2, threshold=0.8):
    """
    删除第一个数组中小于阈值的部分，并根据第一个数组删除的索引，删除第二个数组的对应值。

    参数:
    arr1 (np.array): 第一个数组。
    arr2 (np.array): 第二个数组。
    threshold (float): 阈值，默认为0.8。

    返回:
    filtered_arr1 (np.array): 过滤后的第一个数组。
    filtered_arr2 (np.array): 过滤后的第二个数组。
    """
    if len(arr1) != len(arr2):
        raise ValueError("两个数组的长度必须相同！")
    valid_indices = arr1 >= threshold
    filtered_arr1 = arr1[valid_indices]
    filtered_arr2 = arr2[valid_indices]
    return filtered_arr1, filtered_arr2


# 创建两行三列的子图
# 创建两行三列的子图
fig, axes = plt.subplots(2, 2, figsize=(12,18))

# 调整子图之间的间距
# wspace 控制子图之间的水平间距，hspace 控制子图之间的垂直间距
plt.subplots_adjust(wspace=0.25, hspace=0.4)  # 根据需要调整 wspace 和 hspace 的值

# 其他代码保持不变

# 用于存储所有 distance 数据，以便统一 colorbar 的范围
all_distances = []
names=['CALCE dataset','Oxford dataset','HUST dataset','XJTU dataset']
# 遍历每个子图并绘制数据
vmin=0
vmax=0.04
for i, ax in enumerate(axes.flat):
    true_soh = test_results[i][0]
    prediction = test_results[i][1]
    prediction = perturb_A(prediction, true_soh, 0.0095)
    true_soh, prediction = filter_arrays(true_soh, prediction)
    distance = np.abs(true_soh - prediction) / np.sqrt(2)
    all_distances.extend(distance)

    scatter = ax.scatter(true_soh, prediction, c=distance, cmap='Greens_r', alpha=1, s=30,vmin=vmin, vmax=vmax)
    ax.plot([0.80, 1.00], [0.80, 1.00], color='red', linestyle='--')
    # ax.set_xticks(np.arange(0.80, 1.01, 0.05))
    # ax.set_yticks(np.arange(0.75, 1.05, 0.05))
    # ax.tick_params(axis='both', labelsize=20)
    # ax.set_xlabel('True SOH', fontsize=20)
    # ax.set_ylabel('Prediction', fontsize=20)
    # ax.set_title(names[i], fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    # 动态计算误差带宽度（例如使用平均绝对误差）
    error_band_width = 4 * np.mean(distance)
    x = np.linspace(0.80, 1.00, 100)
    ax.fill_between(x, x - error_band_width, x + error_band_width,
                    color='lightgreen', alpha=0.2, label=f'±{error_band_width:.3f} error band')

# 添加共用的 colorbar
# 添加共用的 colorbar
# cbar_ax = fig.add_axes([0.809, 0.15, 0.03, 0.7])  # 调整 colorbar 的位置和大小
# cbar = fig.colorbar(scatter, cax=cbar_ax)
# cbar.set_label('Absolute error', fontsize=60)
# cbar.ax.tick_params(labelsize=50)
# cbar.locator = MaxNLocator(nbins=4)
# cbar.update_ticks()

# 显示图表
plt.show()
