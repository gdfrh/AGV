import pickle
from Config import *
import plotly.express as px
import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob
import matplotlib.pyplot as plt
from deap import base, creator, tools

# 获取所有pkl文件路径（推荐使用绝对路径）
folder = 'Scatter_plot'
file_paths = glob.glob(os.path.join(folder, '*.pkl'))

# 定义要排除的文件名（更稳定的方式）
exclude_name = 'all_point.pkl'

# 使用文件名比对而非完整路径
file_paths = [f for f in file_paths
             if os.path.basename(f) != exclude_name]

# 验证结果
print("过滤后的文件列表：")
for p in file_paths:
    print(p)
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
    if os.path.basename(file_path) == '0.pkl':
        for order_idx, order_data in enumerate(loaded_data['order']):
            df = pd.DataFrame(order_data)
            save_path = f'results/result_{order_idx}/'
            os.makedirs(save_path, exist_ok=True)
            # 保存为Excel文件
            excel_name = "order.xlsx"
            df.to_excel(os.path.join(save_path, excel_name), index=False)

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
                 hover_data=['agv_distribution',  '组装区', '铸造区', '清洗区', '包装区', '焊接区', '喷漆区',
                             '配置区'], color='group')

# # # 显示图表
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
save_path = 'results/'
os.makedirs(save_path, exist_ok=True)
file_name = 'compare.png'  # 文件名包含案例编号
plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形避免内存泄漏
# ------------------------
#   绘制全局散点图
# 初始化DEAP类型（需放在所有DEAP操作之前）
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # 双目标最小化
creator.create("Individual", list, fitness=creator.FitnessMulti)

# 加载数据
file_path = glob.glob('Scatter_plot/all_point.pkl')
with open(file_path[0], 'rb') as file:
    loaded_data = pickle.load(file)

# 转换为DEAP Individual对象
pop = []
for time, energy in loaded_data:  # 假设数据格式为(time, energy)
    ind = creator.Individual([time, energy])
    ind.fitness.values = (time, energy)  # 设置适应度值
    pop.append(ind)

# 执行非支配排序（只需执行一次）
fronts = tools.sortNondominated(pop, len(pop))

# 提取前沿点坐标
def extract_coordinates(front):
    return np.array([[ind[1], ind[0]] for ind in front])  # 转换为(time, energy)数组

first_front_points = extract_coordinates(fronts[0])
other_points = extract_coordinates([ind for front in fronts[1:] for ind in front])

# 可视化设置
plt.figure(figsize=(12, 8), dpi=100)
ax = plt.gca()

# 绘制散点图（注意坐标轴方向）
ax.scatter(
    x=other_points[:, 1],  # X轴：能耗
    y=other_points[:, 0],  # Y轴：时间
    c='#A9A9A9',
    alpha=0.5,
    edgecolors='b',
    linewidths=0.8,
    label='Other Solutions',
    zorder=2
)

ax.scatter(
    x=first_front_points[:, 1],
    y=first_front_points[:, 0],
    c='#000000',
    alpha=0.9,
    edgecolors='k',
    linewidth=0.5,
    # marker='s',
    s=50,
    label='Pareto Front (1st Front)',
    zorder=3
)

# 坐标轴设置
ax.set_xlabel('Energy Consumption', fontsize=12)
ax.set_ylabel('Completion Time', fontsize=12)
ax.set_title('Pareto Front Visualization', fontsize=14, pad=15)

# === 新增前沿连线代码 ===
if len(first_front_points) > 1:  # 仅当有多个点时绘制连线
    sorted_front = first_front_points[np.argsort(first_front_points[:, 1])]

    ax.plot(
        sorted_front[:, 1],
        sorted_front[:, 0],
        color='#000000',
        linestyle='-',
        linewidth=1.5,
        alpha=0.8,
        label='Pareto Front Line',
        zorder=4
    )

    # 更新图例
    ax.legend(title='Solution Types',
              loc='upper right',
              framealpha=0.9)

plt.tight_layout()
save_path = 'results/'
os.makedirs(save_path, exist_ok=True)
file_name = 'scatter plot.png'  # 文件名包含案例编号
plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形避免内存泄漏
# ------------------------
# 绘制迭代图（横向布局）
file_path = glob.glob('iterations/iterations.pkl')
with open(file_path[0], 'rb') as file:
    loaded_data = pickle.load(file)

for case_idx, case_data in enumerate(loaded_data):
    makespan = sorted(case_data[0], reverse=True)
    energy = sorted(case_data[1], reverse=True)
    num_iterations = len(makespan)
    iterations = np.arange(0, num_iterations)

    # 修改1：调整画布尺寸和子图排列方向
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 改为横向排列
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 修改2：优化第一个子图设置
    ax1.plot(iterations, makespan,
             color='#2E75B6',
             linewidth=2.2,
             label='ALNS')
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('最大完工时间/min', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.8)
    ax1.set_title('最大完工时间优化轨迹', fontsize=13, pad=12)

    # 修改3：优化第二个子图设置
    ax2.plot(iterations, energy,
             color='#C00000',
             linewidth=2.2,
             label='ALNS')
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('机器能耗/kWh', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.8)
    ax2.set_title('机器能耗优化轨迹', fontsize=13, pad=12)

    # 新增：添加公共标题
    fig.suptitle('ALNS优化过程分析',
                 y=0.98,
                 fontsize=15,
                 fontweight='bold')

    # 修改4：统一坐标轴范围设置
    for ax in [ax1, ax2]:
        ax.set_xlim(-0.5, num_iterations + 0.5)
        ax.legend(loc='upper right', framealpha=0.9)

    # 修改5：调整子图间距
    plt.subplots_adjust(wspace=0.15)  # 控制子图横向间距

    # 保存设置保持不变
    save_path = f'results/result_{case_idx}/'
    os.makedirs(save_path, exist_ok=True)
    file_name = 'iteration_horizontal.png'  # 修改文件名以区分布局
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------
# 绘制权重迭代图
file_path = glob.glob('iterations/weights.pkl')
with open(file_path[0], 'rb') as file:
    loaded_data = pickle.load(file)
for case_idx, case_data in enumerate(loaded_data):

    # 转换为NumPy数组方便处理
    weights_array = np.array(case_data)

    # 设置中文显示（如果不需要可删除）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 折线图（推荐首选）
    plt.figure(figsize=(10, 6))

    # 提取每个权重的变化序列
    operator1 = weights_array[:, 0]
    operator2 = weights_array[:, 1]
    operator3 = weights_array[:, 2]

    # 生成迭代次数序列
    iterations = np.arange(len(weights_array))

    # 绘制三条曲线
    plt.plot(iterations, operator1,
             label='随机破坏修复',
             color='#1f77b4',
             # marker='o',
             linestyle='-')

    plt.plot(iterations, operator2,
             label='后悔修复',
             color='#ff7f0e',
             # marker='s',
             linestyle='--')

    plt.plot(iterations, operator3,
             label='贪婪修复',
             color='#2ca02c',
             # marker='^',
             linestyle='-.')

    # 图表装饰
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('权重占比', fontsize=12)
    plt.title('ALNS算子权重变化趋势', fontsize=14, pad=20)
    plt.legend(title='算子类型',
               loc='upper left',
               bbox_to_anchor=(1, 1))  # 图例放在右侧
    plt.grid(True,
             linestyle='--',
             alpha=0.6)
    plt.tight_layout()
    save_path = f'results/result_{case_idx}/'
    os.makedirs(save_path, exist_ok=True)
    file_name = 'weight.png'  # 文件名包含案例编号
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形避免内存泄漏

# # ==================== 数据加载 ====================
# file_path = glob.glob('iterations/iterations_greedy.pkl')
# # 从 pkl 文件加载数据
# with open(file_path[0], 'rb') as file:
#     loaded_data = pickle.load(file)
# time_1 = loaded_data[0]
# energy_1 = loaded_data[1]
#
# file_path = glob.glob('iterations/iterations_regret.pkl')
# # 从 pkl 文件加载数据
# with open(file_path[0], 'rb') as file:
#     loaded_data = pickle.load(file)
# time_2 = loaded_data[0]
# energy_2 = loaded_data[1]
#
# # ==================== 画布设置 ====================
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
# # ==================== 通用参数 ====================
# iterations = np.linspace(0, 500, len(time_1))  # 生成迭代次数
# line_style = {
#     'group1': {'color': '#1f77b4', 'label': '贪婪算法', 'ls': '-', 'marker': 'o'},
#     'group2': {'color': '#ff7f0e', 'label': '后悔算法', 'ls': '--', 'marker': 's'}
# }
#
# # ==================== 完工时间对比 ====================
# ax1.plot(iterations, time_1, **line_style['group1'], lw=1.5, markersize=5)
# ax1.plot(iterations, time_2, **line_style['group2'], lw=1.5, markersize=5)
# ax1.set_ylabel('最大完工时间 (min)', fontsize=12)
# ax1.set_title('多算法优化轨迹对比分析', fontsize=14, pad=20)
# ax1.grid(True, linestyle=':', alpha=0.6)
# ax1.legend(loc='upper right', fontsize=10)
#
# # ==================== 能耗对比 ====================
# ax2.plot(iterations, energy_1, **line_style['group1'], lw=1.5, markersize=5)
# ax2.plot(iterations, energy_2, **line_style['group2'], lw=1.5, markersize=5)
# ax2.set_xlabel('迭代次数', fontsize=12)
# ax2.set_ylabel('综合能耗 (kWh)', fontsize=12)
# ax2.grid(True, linestyle=':', alpha=0.6)
#
# # ==================== 动态坐标范围 ====================
# def auto_scale(data_list):
#     """自动计算多组数据的坐标范围"""
#     dmin = min(np.nanmin(d) for d in data_list)
#     dmax = max(np.nanmax(d) for d in data_list)
#     span = dmax - dmin
#     return (dmin - 0.1*span, dmax + 0.1*span)
#
# # 设置坐标范围
# ax1.set_ylim(auto_scale([time_1, time_2]))
# ax2.set_ylim(auto_scale([energy_1, energy_2]))
# ax2.set_xlim(0, 500)
#
# # # ==================== 样式优化 ====================
# # for ax in [ax1, ax2]:
# #     ax.tick_params(axis='both', labelsize=10)
# #     for spine in ['top', 'right']:
# #         ax.spines[spine].set_visible(False)
#
# plt.tight_layout()
# plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
# plt.show()



