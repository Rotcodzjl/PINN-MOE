import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
def plot_model_comparison(model_data, y_range=[50, 80], x_range=[0, 50], size_interval=40):
    """
    绘制模型性能对比图

    参数:
        model_data: 嵌套列表格式 [['model_name', accuracy, speed, model_size, has_outer_ring, ring_size], ...]
        y_range: 纵坐标范围 [最小值, 最大值]
        x_range: 横坐标范围 [最小值, 最大值]
        size_interval: 五个参考球的大小间隔（控制大小差异）
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    # ax.set_aspect('equal', adjustable='box')  # 确保圆形不被拉伸

    # 计算坐标轴的物理比例（数据单位与显示单位的比例）
    fig_width, fig_height = plt.gcf().get_size_inches()
    data_width = x_range[1] - x_range[0]
    data_height = y_range[1] - y_range[0]
    x_to_display = fig_width / data_width
    y_to_display = fig_height / data_height
    aspect_ratio = y_to_display / x_to_display

    # 绘制每个模型
    for i, model in enumerate(model_data):
        name = model[0]
        acc = model[1]
        speed = model[2]
        size = model[3]
        has_ring = model[4]
        ring_size = model[5] if has_ring else 0

        # 计算内圆半径（考虑坐标轴比例）
        inner_area = size
        base_radius = math.sqrt(inner_area / math.pi)
        inner_radius = base_radius * aspect_ratio  # 根据纵横比调整半径
    # 创建颜色映射
    # colors = plt.cm.tab20(np.linspace(0, 1, len(model_data)))
    custom_colors = [
        '#DC143C', '#1f77b4', '#ff7f0e', '#2ca02c',
        '#9467bd', '#8c564b', '#e377c2', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', # 这里排除灰色
        '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        '#7f7f7f',
    ]
    colors = [custom_colors[i % len(custom_colors)] for i in range(len(model_data))]
    # plt.gca().set_aspect('equal', adjustable='box')  # 关键修改
    # 绘制每个模型
    for i, model in enumerate(model_data):
        name = model[0]
        acc = model[1]
        speed = model[2]
        size = model[3]
        has_ring = model[4]
        ring_size = model[5] if has_ring else 0

        # 计算内圆半径 (面积=πr²)
        inner_area = size
        inner_radius = math.sqrt(inner_area / math.pi)

        # 处理外环
        if has_ring:
            total_area = inner_area + ring_size
            outer_radius = math.sqrt(total_area / math.pi)
            ring_width = outer_radius - inner_radius
        else:
            outer_radius = inner_radius
            ring_width = 0
        alpha=2
        # 绘制带外环的圆盘
        # circle = plt.Circle((speed, acc), outer_radius,
        #                     color=colors[i], alpha=0.3, fill=True)
        if name=='Proposed model':
            ellipse = Ellipse(xy=(speed, acc), width=alpha * inner_radius * aspect_ratio, height=alpha * inner_radius,
                              angle=0, edgecolor='b', facecolor='red', alpha=0.3,)
        else:
            ellipse = Ellipse(xy=(speed, acc), width=alpha*inner_radius*aspect_ratio, height=alpha*inner_radius,
                              angle=0, edgecolor='b', facecolor=colors[i], alpha=0.3)
        ellipse2 = Ellipse(xy=(speed, acc), width=alpha * 0.03 * aspect_ratio, height=alpha * 0.03,
                          angle=0, edgecolor='b', facecolor='white', alpha=0.3)
        plt.gca().add_patch(ellipse)
        plt.gca().add_patch(ellipse2)

        # 绘制内圆（颜色较深）
        # inner_circle = plt.Circle((speed, acc), inner_radius,
        #                           color=colors[i], alpha=0.7, fill=True)
        # plt.gca().add_patch(inner_circle)
        plt.text(speed-0*alpha*inner_radius*aspect_ratio, acc, name, ha='center', va='center', fontsize=10)

    # 设置坐标轴
    plt.xlabel('M-FLOPs', fontsize=16)
    plt.ylabel('RMSE(%)', fontsize=16)
    # plt.title('Deep Learning Model Performance Comparison', fontsize=18)
    plt.xlim(x_range[0], x_range[1])
    plt.xticks(fontsize=14)
    plt.ylim(y_range[0], y_range[1])
    plt.yticks(fontsize=14)

    # 添加图例
    legend_elements = []
    for i, model in enumerate(model_data):
        legend_elements.append(plt.Line2D([0], [0],
                                          marker='o',
                                          color='w',
                                          label=model[0],
                                          markerfacecolor=colors[i],
                                          markersize=10))

    plt.legend(handles=legend_elements, bbox_to_anchor=(0.75, 1), loc='upper left',frameon=False)
    # 添加半径参考球（右下角水平排列，只显示5个）
    max_speed = x_range[1]
    min_acc = y_range[0]

    # 计算5个参考球的大小（基于size_interval参数）
    min_size = min(model[3] for model in model_data)
    max_size = max(model[3] for model in model_data)

    # 生成5个等间隔的大小值
    size_values = np.linspace(min_size, max_size, 5)
    if size_interval > 0:
        # 如果指定了size_interval，则使用固定间隔
        size_values = [min_size + i * size_interval for i in range(5)]

    rect = Rectangle((7.5,3.2), 15, 1.8,
                     linewidth=1, edgecolor='black', facecolor='none', linestyle='-')
    # 添加到坐标轴
    ax.add_patch(rect)
    # 计算参考球的位置（右下角水平排列）
    start_x = 14.5 # 从x轴的60%位置开始
    y_pos = 4.2 # y轴位置在最低准确率下方
    plt.text(8.5,4.17,"model size(M)->",fontsize=12)
    # 计算间距（根据x轴范围调整）
    # spacing = (max_speed * 0.2) / 10
    spacing=0
    size_values=[0.2,0.5,0.8,1.1,1.4]
    for i, size in enumerate(size_values):
        radius = math.sqrt(size / math.pi)
        x_pos = start_x + i * spacing * 3  # 乘以3增加间距

        # 绘制参考球
        # ref_circle = plt.Circle((x_pos, y_pos), radius,
        #                         color='gray', alpha=0.5)
        ref_ellipse = Ellipse(xy=(x_pos, y_pos), width=alpha*radius*aspect_ratio, height=alpha*radius,
                          angle=0, edgecolor='b', facecolor='gray', alpha=0.5)
        plt.gca().add_patch(ref_ellipse)

        # 添加半径标签
        plt.text(x_pos, y_pos - radius - (max_speed * 0.001),
                 f"{size_values[i]:.1f}", ha='center', va='top', fontsize=8)
    plt.plot([10,10], [0,3], color='black',linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.text(2.05,2.97, "Simple sensor data processing",fontsize=14)
    plt.text(10.5, 2.97, "Embedded lightweight processor", fontsize=14)
    plt.tight_layout()
    plt.show()


# 示例数据 - 根据您提供的图片信息创建
model_examples = [
    ['Proposed model', 0.529, 3.4, 0.237, False, 0],
    # ['Attention-MoE', 0.7721, 3.2, 0.2232, False, 0],
    ['MLP', 1.37, 2.67, 0.624, False, 0],
    ['CNN', 1.312, 7.08, 0.394, False, 0],
    ['LSTM', 2.138, 12.65, 0.625553, False, 0],
    ['GRU', 2.338, 9.65, 0.46921, False, 0],
    ['Bi-LSTM', 1.6721, 15.53, 0.7332, False, 0],
    ['CNN-LSTM', 1.12, 8.83, 0.45623, False, 0],
    ['CNN-GRU', 1.21, 7.83, 0.43623, False, 0],
    ['TPA-CNN-LSTM', 0.9724, 17.16, 0.4436, False, 0],
    ['Transformer', 1.552, 18.37, 1.3542, False, 0],
    ['CNN-Transformer', 1.294, 15.68, 1.3672, False, 0],


]

# 使用新参数调用函数
plot_model_comparison(
    model_data=model_examples,
    y_range=[0,5],  # 纵坐标范围
    x_range=[1, 22],  # 横坐标范围
    size_interval=0  # 五个球的大小间隔
)