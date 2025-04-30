import matplotlib.pyplot as plt
import numpy as np

# ----------- 中文显示配置 -----------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False
# ----------------------------------

# 数据准备
instances = ['实例1', '实例2', '实例3', '实例4', '实例5']
legends = ['NSGA2', 'NSGA3', 'SPEA2', 'Random']
colors = ['#9bbf8a', '#82afda', '#f79059', '#e7dbd3']  # 使用前4个颜色

# 原始数据
data_time = np.array([
    [1641, 2708, 2108, 2187],
    [2472, 4930, 6002, 4356],
    [4535, 4902, 1858, 7891],
    [5762, 5199, 5523, 11083],
    [6733, 6649, 11347, 6642]
])

data_energy = np.array([
    [11698, 27013, 11886, 18581],
    [40285, 32375, 21699, 40259],
    [28240, 31641, 25831, 39719],
    [55364, 55492, 42669, 65733],
    [59191, 77105, 49810, 72926]
])


# 数据归一化函数
def normalize_data(data, new_min=0.1, new_max=0.9):
    """弹性范围归一化，默认映射到[0.1, 0.9]区间"""
    data_min = np.min(data)
    data_max = np.max(data)
    # 防止除零错误
    range_adjust = 1e-6 if data_max == data_min else (data_max - data_min)
    # 线性映射到新区间
    return (data - data_min) / range_adjust * (new_max - new_min) + new_min


# 归一化处理
norm_time = normalize_data(data_time)
norm_energy = normalize_data(data_energy)

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


# 通用绘图函数
def plot_bars(ax, data, title):
    bar_width = 0.2  # 调整柱子宽度
    x = np.arange(len(instances))

    for i in range(len(legends)):
        offset = bar_width * (i - len(legends) / 2 + 0.5)
        ax.bar(x + offset, data[:, i],
               width=bar_width,
               label=legends[i],
               color=colors[i],
               edgecolor='black',
               linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=0, ha='center')
    ax.set_xlabel('实例类型', fontsize=12)
    ax.set_ylabel('归一化方差值', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)  # 统一Y轴范围


# 绘制时间方差图
plot_bars(ax1, norm_time, '时间方差对比')

# 绘制能耗方差图
plot_bars(ax2, norm_energy, '能耗方差对比')

# 统一图例
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels,
           title='算法',
           bbox_to_anchor=(0.92, 0.95),
           ncol=1,
           fontsize=10)

# 调整布局
plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为图例预留空间
plt.show()