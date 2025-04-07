import pickle
from Config import *
import plotly.express as px
import pandas as pd
import glob
import matplotlib.pyplot as plt

# 替换 'folder_name' 为你目标文件夹的路径
file_paths = glob.glob('Scatter_plot/0.pkl')

# 用于存储所有的数据
combined_data = {
    'energy': [],
    'time': [],
    'agv_distribution': [],
    'order': [],
    'distributions_dict': [],
    'group': [],  # 用于存储每个数据来源的 group 信息
    'point_id': []  # 用于存储每个点的唯一标识符（x轴）
}

arm_distributions = []
point_counter = 0  # 用于创建唯一的点标识符
# 遍历所有文件
for idx, file_path in enumerate(file_paths):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    # 获取文件中的数据
    energy_pic = loaded_data['energy']
    time_pic = loaded_data['time']
    agv_distributions = loaded_data['agv_distribution']
    order_distributions = loaded_data['order']
    distributions_dicts = loaded_data['distributions_dict']

    # 为每个文件的数据添加一个group字段
    group_label = f'Algorithm {idx}'  # 给每个文件添加一个独特的标识符
    combined_data['energy'] += energy_pic
    combined_data['time'] += time_pic
    combined_data['agv_distribution'] += agv_distributions
    combined_data['order'] += order_distributions
    combined_data['group'] += [group_label] * len(energy_pic)  # 对应的每个数据点分配相同的 group 标签
    combined_data['point_id'] += [point_counter + i for i in range(len(energy_pic))]  # 为每个点分配唯一标识符

    # 处理 distributions_dicts，可能是字典格式的，转化为可展平的格式
    for dic in distributions_dicts:
        arm_distributions.append(dic)

    point_counter += len(energy_pic)  # 更新计数器，确保每个点有唯一的标识符

# 计算最大长度
max_length = max(len(lst) for lst in combined_data.values())

# 调整所有列表到相同的长度
for key in combined_data:
    current_length = len(combined_data[key])
    if current_length < max_length:
        # 补全列表
        combined_data[key].extend([None] * (max_length - current_length))

# 将数据转换为 DataFrame
df = pd.DataFrame(combined_data)

# 将字典展开成 DataFrame
df_dicts = pd.DataFrame(arm_distributions)

# 合并 DataFrame
df_final = pd.concat([df, df_dicts], axis=1)

# 使用 Plotly Express 创建散点图
fig = px.scatter(df_final, x='energy', y='time', title="Energy vs. Time Scatter Plot",
                 hover_data=['agv_distribution', '组装区', '铸造区', '清洗区', '包装区', '焊接区', '喷漆区',
                             '配置区'], color='group')

# 显示图表
fig.show()

# 定义图例映射关系
legend_mapping = {
    0: "NSGA-II_ALNS",
    1: "NSGA-II_Random",
    2: "Random_ALNS",
    3: "Random_Random",
    4: "NSGA-II_ALNS-greedy",
    5: "NSGA-II_ALNS-regret",
    6: "NSGA-III_ALNS"
}

# 数据清洗
df_clean = df_final.dropna(subset=['energy', 'time'])

# 创建图形
plt.figure(figsize=(12, 8), dpi=100)
ax = plt.gca()

# 颜色列表（保持与plotly默认颜色一致）
colors = px.colors.qualitative.Plotly

# 为每个算法组绘制散点
groups = sorted(df_clean['group'].unique())
for idx, group in enumerate(groups):
    subset = df_clean[df_clean['group'] == group]

    # 获取对应的图例标签
    label = legend_mapping.get(idx, group)  # 找不到映射时显示原始组名

    plt.scatter(
        subset['energy'],
        subset['time'],
        alpha=0.7,
        edgecolors='w',
        linewidths=0.5,
        color=colors[idx % len(colors)],  # 循环使用颜色
        label=label,
        zorder=2  # 确保点在上层
    )

# 坐标轴设置
plt.xlabel('Energy Consumption', fontsize=12, labelpad=10)
plt.ylabel('Completion Time', fontsize=12, labelpad=10)
plt.title('Energy vs. Time Distribution', fontsize=14, pad=20)

# 网格设置
ax.grid(True,
        linestyle='--',
        alpha=0.6,
        zorder=1)  # 网格在下层

# 图例设置
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    title='Algorithm Versions',
    bbox_to_anchor=(1.05, 0.95),  # 调整图例位置
    loc='upper left',
    frameon=True,
    framealpha=0.9,
    fontsize=10
)

# 调整边距
plt.subplots_adjust(right=0.75)  # 为图例留出空间

# 显示图表
plt.tight_layout()
plt.show()


# # 从文件加载数据
# with open(f'{compare}.pkl', 'rb') as file:
#     loaded_data = pickle.load(file)
#
# # 获取保存的数据
# energy_pic = loaded_data['energy']
# time_pic = loaded_data['time']
# agv_distributions = loaded_data['agv_distribution']
# # order_distributions = loaded_data['order']
# distributions_dicts = loaded_data['distributions_dict']
#
# # 计算最大长度
# max_length = max(len(lst) for lst in loaded_data.values())
#
# # 调整所有列表到相同的长度
# for key in loaded_data:
#     current_length = len(loaded_data[key])
#     if current_length < max_length:
#         # 补全列表
#         loaded_data[key].extend([None] * (max_length - current_length))
# # 将字典转换为 DataFrame
# df = pd.DataFrame(loaded_data)
# df_dicts = pd.DataFrame(distributions_dicts)
# # 合并 DataFrame
# df_final = pd.concat([df, df_dicts], axis=1)
# # 使用 Plotly Express 创建散点图
# fig = px.scatter(df_final, x='energy', y='time', title="Energy vs. Time Scatter Plot",
#                  hover_data=['agv_distribution', '组装区', '铸造区', '清洗区', '包装区', '焊接区', '喷漆区',
#                              '配置区'])

