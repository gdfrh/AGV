from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib as mpl
from Config import *
import numpy as np
import pickle
import glob
import os


def crowding_distance(points):
    # 对时间和能耗分别排序
    time_sorted = sorted(points, key=lambda x: x[0])  # 时间排序
    energy_sorted = sorted(points, key=lambda x: x[1])  # 能耗排序

    # 初始化拥挤度
    crowding = np.zeros(len(points))

    # 对每个目标计算拥挤度
    for i, sorted_points in enumerate([time_sorted, energy_sorted]):
        # 对目标进行排序
        for j in range(1, len(points) - 1):
            # 时间的最大最小值处理
            if i == 0:
                left = sorted_points[j - 1][0]
                right = sorted_points[j + 1][0]
                max_time = max([p[0] for p in points])
                min_time = min([p[0] for p in points])
                crowding[j] += (right - left) / (max_time - min_time)
            # 能耗的最大最小值处理
            if i == 1:
                left = sorted_points[j - 1][1]
                right = sorted_points[j + 1][1]
                max_energy = max([p[1] for p in points])
                min_energy = min([p[1] for p in points])
                crowding[j] += (right - left) / (max_energy - min_energy)
    return crowding


# 替换 'folder_name' 为你目标文件夹的路径
file_paths = glob.glob('Pareto_Crowding_Distance/*.pkl')
# 用于存储不同前缀的文件数据
Pareto_Crowding_Distances = {str(i): [] for i in range(compare_number)}  # 创建一个字典，键是字符串 '0' 到 '7'
# 遍历所有文件
for idx, file_path in enumerate(file_paths):
    with open(file_path, 'rb') as file:
        # 用于存储所有的数据
        points = []
        loaded_data = pickle.load(file)
    energy = loaded_data['energy']
    time = loaded_data['time']
    # 用于存储所有的数据
    points = [(energy[i], time[i]) for i in range(len(energy))]
    # 计算每个解的拥挤度
    crowding_values = crowding_distance(points)
    # 提取文件名
    file_name = os.path.basename(file_path)
    # 提取文件名前缀（假设前缀为文件名的开头部分，数字在 '_' 前）
    prefix = file_name.split('_')[0]

    # 判断是否是数字前缀（0 到 compare_number），如果是，将其添加到相应的列表中
    if prefix.isdigit() and int(prefix) in range(compare_number):
        Pareto_Crowding_Distances[prefix].append(crowding_values)
# for key, value in Pareto_Crowding_Distances.items():
#     for i in range(len(value)):
#         print(Pareto_Crowding_Distances[key][i])
# 创建颜色映射
keys = list(Pareto_Crowding_Distances.keys())
cmap = mpl.colormaps['tab20']  # 使用新的方式获取 'tab20' 颜色映射
color_dict = {key: cmap(i/len(keys)) for i, key in enumerate(keys)}
# 合并并过滤数据
plot_data = []
for key in keys:
    values = []

    for arr in Pareto_Crowding_Distances[key]:
        filtered = [v for v in arr if v != 0]  # 过滤零值
        if not filtered:  # 跳过空数据
            print(f"警告: 算法{key} 的一个数组被过滤后为空")
            continue
        values.extend(filtered)

    if not values:  # 处理整个键无数据的情况
        print(f"严重警告: {key} 无有效数据，已跳过")
        continue
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten()
    plot_data.append((key, scaled_values))

# ================= 可视化设置 =================
plt.figure(figsize=(16, 8))
ax = plt.gca()

# 柱状图参数
bar_width = 0.8
group_padding = 2  # 组间间隔
# ================= 绘制柱状图 =================
current_pos = 0
x_ticks = []
x_labels = []

for key, values in plot_data:
    # 生成x轴位置
    x_positions = np.arange(current_pos, current_pos + len(values))

    # 绘制条形
    bars = ax.bar(
        x_positions,
        values,
        width=bar_width,
        color=color_dict[key],
        edgecolor='white',
        alpha=0.9,
        label=key
    )
    # 添加数据标签
    for idx, rect in enumerate(bars):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.,
            height + 0.05,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=45
        )

    # 记录坐标轴标签
    x_ticks.append(np.median(x_positions))
    x_labels.append(key)

    current_pos += len(values) + group_padding
# ================= 图表装饰 =================
# 坐标轴设置
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_ylabel('Value')
ax.set_title('Process Parameters Distribution', pad=20)

# 网格线设置
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 自定义图例
legend_elements = [Patch(facecolor=color_dict[key], label=key) for key in keys]
ax.legend(
    handles=legend_elements,
    title='Process Types',
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

# 优化布局
plt.tight_layout()
plt.show()
