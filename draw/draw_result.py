import matplotlib.pyplot as plt
import numpy as np

def plot_rmse_bar_groups(group1, group2, group3, group4, labels, model_colors=None):
    """
    绘制四组 RMSE 值的水平柱状图，每组柱状图挨着，不同组分开，
    且每个模型在所有组中颜色相同，但不同模型颜色不同。

    参数:
    group1, group2, group3, group4: 每组四个神经网络的 RMSE 值（列表或数组）。
    labels: 每组数据的标签（列表，长度为 4）。
    model_colors: 可选，每个模型的颜色（列表，长度为 4）。
    """
    # 确保每组数据有四个值
    assert len(group1) == len(group2) == len(group3) == len(group4) == 4, "每组数据必须包含四个值"

    # 设置柱状图的宽度和间距
    bar_width = 0.2  # 每个柱状图的宽度
    group_spacing = 1.0  # 不同组之间的间距

    # 计算每组柱状图的位置
    x = np.arange(len(group1) * 4)  # 每组有四个柱状图，共四组
    group_positions = [
        x[i * len(group1): (i + 1) * len(group1)] + i * group_spacing
        for i in range(4)
    ]

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 使用科研绘图配色（如 Nature 或 Science 期刊常用配色）
    if model_colors is None:
        model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 每个模型的颜色
    models=['proposed','No_phisics','MLP','CNN']
    # 绘制每组柱状图
    for i in range(4):  # 遍历每组
        for j in range(4):  # 遍历每个模型
            plt.bar(group_positions[i][j], [group1[j], group2[j], group3[j], group4[j]][i],
                    width=bar_width, color=model_colors[j], edgecolor='black',
                    label=models[j])  # 只在第一组添加图例
    # 在每个模型的四组 RMSE 值之间绘制连接线
    for j in range(4):  # 遍历每个模型
        x_points = [group_positions[i][j] for i in range(4)]  # 每个模型的 x 坐标
        y_points = [group1[j], group2[j], group3[j], group4[j]]  # 每个模型的 y 值
        plt.plot(x_points, y_points, color=model_colors[j], linestyle='--', linewidth=1, alpha=0.7)
    # 设置横坐标标签
    xticks = []
    xtick_labels = []
    for i in range(4):
        xticks.extend(group_positions[i])
        # xtick_labels.extend([f'Model {j + 1}' for j in range(4)])
    plt.xticks(xticks, xtick_labels, rotation=45, ha='right')

    # 添加标题和标签
    # plt.title('RMSE Loss for CACLE dataset', fontsize=14, pad=20)
    plt.xlabel('(d) HUST dataset', fontsize=16)
    plt.ylabel('RMSE Loss(%)', fontsize=16)

    # 添加图例
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))  # 去重
    plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

    # 设置网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 美化图形
    plt.tight_layout()
    plt.show()

#cacle
# group1 = [0.44769,0.45732,0.61730,1.1632]  # 训练集最大
# group2 = [0.45596,0.64317,0.72351,1.4523]  # 训练集中等
# group3 = [0.45695,0.87542,0.95132,1.6241]  # 训练集较小
# group4 = [0.45684,1.1321,1.2475,1.9742]  # 训练集最小
#xjtu
# group1 = [1.2280,1.3581,2.7149,2.6149]  # 训练集最大
# group2 = [1.2465,1.4782,3.2516,3.5432]  # 训练集中等
# group3 = [1.2782,2.2571,4.7205,4.9887]  # 训练集较小
# group4 = [1.3542,4.3256,6.2572,7.4125]  # 训练集最小
#ox
# group1 = [0.91321,0.9546,1.1723,1.0321]  # 训练集最大
# group2 = [0.93835,1.1458,1.3667,1.1274]  # 训练集中等
# group3 = [0.96645,1.2477,1.6742,1.2672]  # 训练集较小
# group4 = [1.12312 ,1.4362,2.3142,1.4653]  # 训练集最小
#hust
group1 = [2.1132, 2.1363,2.3721, 2.1415 ]  # 训练集最大
group2 = [2.2536,2.6482,3.4635,3.6781]  # 训练集中等
group3 = [2.4563,3.7156,4.5123,	5.6231]  # 训练集较小
group4 = [2.6312,4.8264,6.2781,	6.4529]  # 训练集最小

# 每组数据的标签
labels = ['Large Training Set', 'Medium Training Set', 'Small Training Set', 'Smallest Training Set']

# 调用函数绘图
plot_rmse_bar_groups(group1, group2, group3, group4, labels)