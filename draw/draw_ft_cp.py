import matplotlib.pyplot as plt
import numpy as np


def plot_performance_comparison():
    # 设置数据
    models = ['Proposed direct-training','Proposed fine-tuning', 'MLP direct-training', 'CNN direct-training',
              'LSTM direct-training']
    datasets = ['CACLE dataset', 'Oxford dataset', 'HUST dataset', 'XJTU dataset',]

    # 根据图片中的数据设置（注意：部分数据在图片中不完全清晰）
    data = np.array([
        # MMLU-Pro (EM)
        [0.62,1.71, 3.24,2.85, 3.15],
        # GPQA-Diamond (Pass@1)
        # MAHI 500 (EM)
        [0.75,1.46, 2.61, 1.96, 2.43],
        [1.75,2.73, 6.25, 7.41, 6.54],
        # AIME 2024 (Pass@1)
        [1.16,1.62, 6.27, 6.45, 4.83],
        # Codeforces (Percentile)
    ])

    # 设置图形
    plt.figure(figsize=(10, 7))

    # 使用与原图相同的配色方案
    colors = ['#004080', '#b3b3b3', '#ffe699', '#e6e6ff', '#9467bd', '#8c564b']

    # 设置柱状图宽度和位置
    bar_width = 0.12
    x = np.arange(len(datasets))

    # 绘制每个模型的柱状图
    for i, model in enumerate(models):
        plt.bar(x + i * bar_width, data[:, i], width=bar_width,
                color=colors[i], label=model)

    # 设置图表属性
    # plt.title('Model Performance Comparison Across Tasks', pad=20,fontsize=20)
    plt.ylabel('RMSE (%)',fontsize=20)
    plt.xticks(x + bar_width * 1.5, datasets, rotation=0,fontsize=16)
    plt.ylim(0, 10)

    # 添加数据标签（仅显示主要模型）
    for i, dataset in enumerate(datasets):
        for j in range(5):  # 只标注前4个主要模型
            height = data[i, j]
            plt.text(x[i] + j * bar_width, height + 0.1, f'{height:.1f}',
                     ha='center', va='bottom', fontsize=12)

    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, 0.98),
              frameon=True, framealpha=1, edgecolor='black',fontsize=16)

    # 添加网格线
    plt.grid(axis='y', alpha=0.3)
    plt.yticks([0, 2.5,5,7.5, 10], fontsize=18)
    # 调整布局
    plt.tight_layout()
    plt.show()


# 调用函数
plot_performance_comparison()