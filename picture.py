import matplotlib.pyplot as plt
import pickle
import glob
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib import rcParams
from Config import *
from matplotlib.patches import Patch
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import rcParams
from collections import defaultdict
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os

# 设置 matplotlib 使用中文字体
rcParams['font.family'] = 'SimHei'  # 使用黑体（SimHei）字体，或根据需要选择其他字体

# 如果需要支持更好的中文显示，可能还需要设置 'font.sans-serif'
rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei' 等

# 颜色生成函数
def generate_colors(n):
    return plt.cm.tab20(range(n))

color_map = {
    0: '#9bbf8a',  # 订单 0 使用橙色
    1: '#82afda',  # 订单 1 使用蓝色
    2: '#f79059',  # 订单 2 使用绿色
    3: '#e7dbd3',  # 订单 3 使用橙色
    4: '#c2bdde',  # 订单 4 使用蓝色
    5: '#8dcec8',  # 订单 5 使用绿色
    6: '#add3e2',  # 订单 6 使用橙色
    7: '#3480b8',  # 订单 7 使用蓝色
    8: '#fa8878',  # 订单 8 使用绿色
    9: '#ffbe7a',  # 订单 9 使用橙色
    10: '#8a6eaf', # 浅薰衣草紫（补充紫色系）
    11: '#e69aaf', # 玫瑰粉（增加暖色层次）
    12: '#a0a97f', # 橄榄绿（大地色系过渡）
    13: '#5a9b8f', # 深青瓷色（增加冷色深度）
    14: '#c8a2c8', # 淡紫丁香（柔和中性色）
}

#  AGV甘特图
# 修正文件路径中的空格
file_path = glob.glob('Gantt_Chart/agv_timeline_history.pkl')
# 从 pkl 文件加载数据
with open(file_path[0], 'rb') as file:
    loaded_datas = pickle.load(file)

    # 从 pkl 文件加载数据
for load_idx, loaded_data in enumerate(loaded_datas):
    agv_path = glob.glob('Scatter_plot/0.pkl')
    with open(agv_path[0], 'rb') as file:
        agv_data = pickle.load(file)
        agv_distribution = agv_data['agv_distribution']
        # 生产区的小车数量列表
        production_zone_sizes = agv_distribution[load_idx]
# 初始化列存储
    num_columns = len(loaded_data[0])
    columns = [[] for _ in range(num_columns)]
    for row in loaded_data:
        for col_idx, value in enumerate(row):
            columns[col_idx].append(value)
    results = []
    for i in range(num_columns):
        # 数据过滤
        filtered = [x for x in columns[i] if x != 0]
        # 分组处理（每3个元素一组）
        car_tasks = []
        for j in range(0, len(filtered) - 2, 3):
            transport_start = filtered[j]
            transport_end = filtered[j + 2]
            if filtered[j + 1] == -1:
                order_number = 0
            else:
                order_number = filtered[j + 1]

            car_tasks.append({
                'transport': (transport_start, transport_end),
                'order_number': order_number
            })
        results.append(car_tasks)

    # 创建绘图画布
    fig, ax = plt.subplots(figsize=(16, 10))
    # 计算生产区的分界线
    zone_boundaries = []
    current_position = 0
    for size in production_zone_sizes:
        current_position += size
        zone_boundaries.append(current_position)
    # 数据解析优化
    for agv_idx, tasks in enumerate(results):
        for task_idx, task in enumerate(tasks):
            start, end = task['transport']
            order_num = task['order_number']
            # 绘制运输任务
            ax.barh(
                y=agv_idx,
                width=end - start,
                left=start,
                height=0.6,
                color=color_map[order_num],
                edgecolor='black',
                alpha=0.9
            )

            # 添加任务标签
            ax.text(
                x=(start + end) / 2,
                y=agv_idx,
                s=f'订单{order_num+1}\n{end - start:.1f}s',
                ha='center',
                va='center',
                color='black',
                fontsize=8
            )
            # # 包含开始时间和结束时间
            # if end - start > 10:
            #     ax.text(
            #         x= start + 5,
            #         y= agv_idx - 0.42,
            #         s=f'{start:.1f}',
            #         ha='center',
            #         va='center',
            #         color='black',
            #         fontsize=8
            #     )
            #     ax.text(
            #         x= end - 5,
            #         y= agv_idx - 0.42,
            #         s=f'{end:.1f}',
            #         ha='center',
            #         va='center',
            #         color='black',
            #         fontsize=8
            #     )
            # elif end - start <= 10:
            #     ax.text(
            #         x= start + 1.5,
            #         y= agv_idx - 0.42,
            #         s=f'{start:.1f}',
            #         ha='center',
            #         va='center',
            #         color='black',
            #         fontsize=8
            #     )
            #     ax.text(
            #         x= end - 1.5,
            #         y= agv_idx - 0.42,
            #         s=f'{end:.1f}',
            #         ha='center',
            #         va='center',
            #         color='black',
            #         fontsize=8
            #     )
    # 坐标轴优化
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([f'AGV {i+1}' for i in range(len(results))])
    ax.set_xlabel('时间（秒）', fontsize=12)
    ax.set_title('AGV运输任务甘特图', fontsize=14, pad=20)

    # 添加网格线
    ax.grid(axis='x', alpha=0.4, linestyle='--')

    # 创建图例
    legend_handles = [
        mpatches.Patch(color=color_map[i], label=f'订单 {i+1}')
        for i in range(len(color_map))
    ]
    ax.legend(
        handles=legend_handles,
        title='订单颜色映射',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize=12,  # 字体大小
        handlelength=3.5,  # 图例标签的长度
        labelspacing=1.5  # 图例标签之间的间距
    )
    # 绘制生产区分界虚线
    for boundary in zone_boundaries:
        ax.axhline(y=boundary - 0.5, color='gray', linestyle='--', linewidth=1)

    # 绘制生产区名称
    for idx, boundary in enumerate(zone_boundaries):

        if idx == 0:
            # 第一个标签，放在第一个虚线和边框之间
            middle_y = (boundary / 2) - 0.5  # 调整位置

        elif idx == len(zone_boundaries) - 1:
            middle_y = (len(results) + zone_boundaries[idx - 1]) / 2 - 0.5

        else:
            # 后续标签，放在虚线之间
            middle_y = (boundary + zone_boundaries[idx - 1]) / 2 - 0.5
        # 在虚线中间添加生产区名称
        ax.text(
            x=0,  # x 位置不重要，可以调整
            y=middle_y,
            s=work_name[idx],
            ha='center',
            va='center',
            color='black',
            fontsize=10,
            rotation=90  # 旋转文本垂直显示
        )

    # 布局优化
    plt.xlim(0, max([task['transport'][1] for agv in results for task in agv]))
    plt.ylim(-0.5, len(results) - 0.5)
    plt.gca().invert_yaxis()  # 保持AGV 0在顶部
    plt.tight_layout()
    save_path = f'results/result_{load_idx}/'
    os.makedirs(save_path, exist_ok=True)
    file_name = 'agv_gantt.png'  # 文件名包含案例编号
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形避免内存泄漏
# ------------------------
# 订单甘特图
# 找到处理订单部分

file_path = glob.glob('Gantt_Chart/timeline_history.pkl')
# 从 pkl 文件加载数据
with open(file_path[0], 'rb') as file:
    loaded_datas = pickle.load(file)

results = []
for load_idx, loaded_data in enumerate(loaded_datas):
    loaded_data = [row for row in loaded_data if not isinstance(row, tuple)]
    # 初始化列存储
    num_columns = len(loaded_data[0])
    columns = [[] for _ in range(num_columns)]
    for row in loaded_data:
        for col_idx, value in enumerate(row):
            columns[col_idx].append(value)
    result = []
    for i in range(num_columns):
        # 数据过滤
        filtered = [x for x in columns[i] if x != 0 and x != float('inf')]
        # 去除连续重复值
        filtered_clean = []
        prev = None
        for x in filtered:
            if x != prev:
                filtered_clean.append(x)
                prev = x
        cope_times = []

        # 如果列表长度为奇数，在第一位插入0
        if len(filtered_clean) % 2 != 0:
            filtered_clean.insert(0, 0)
        # 将列表每两个值分为一组，前一个为开始时间，后一个为结束时间
        cope_with_orders = []
        cope_times = [(filtered_clean[i], filtered_clean[i + 1]) for i in range(0, len(filtered_clean), 2)]

        for x in range(len(cope_times)):
            cope_with_orders.append({
                'start': cope_times[x][0], 'end': cope_times[x][1]
            })
        result.append(cope_with_orders)
    results.append(result)


# 找到小车运输状态
file_path = glob.glob('Gantt_Chart/timeline_history_1.pkl')

# 从 pkl 文件加载数据
with open(file_path[0], 'rb') as file:
    loaded_datas = pickle.load(file)
results_1 = []
for load_idx, loaded_data in enumerate(loaded_datas):
    # 初始化列存储
    num_columns = len(loaded_data[0])
    columns = [[] for _ in range(num_columns)]
    for row in loaded_data:
        for col_idx, value in enumerate(row):
            columns[col_idx].append(value)
    result_1 = []
    for i in range(num_columns):
        # 数据过滤
        filtered = [x for x in columns[i] if x != 0 and x != float('inf')]
        # 去除连续重复值
        filtered_clean = []
        prev = None
        for x in filtered:
            if x != prev:
                filtered_clean.append(x)
                prev = x
        transport_times = []

        # 遍历列表
        j = 0
        while j < len(filtered_clean):
            if filtered_clean[j] == -1:
                # 检查接下来的两个数
                if j + 2 < len(filtered_clean) and filtered_clean[j + 1] not in [-1, -2, -3] and filtered_clean[j + 2] not in [
                    -1, -2, -3]:
                    start_time = filtered_clean[j + 1]
                    end_time = filtered_clean[j + 2]
                    transport_times.append((start_time, end_time))
                    j += 3  # 跳过已处理的 3 个元素
                else:
                    j += 1  # 跳过当前 -1
            else:
                j += 1  # 继续检查下一个元素
        agv_transport = []
        for x in range(len(transport_times)):
            agv_transport.append({
                'start': transport_times[x][0], 'end': transport_times[x][1]
            })
        result_1.append(agv_transport)
    results_1.append(result_1)

# 先找到等待小车状态
file_path = glob.glob('Gantt_Chart/timeline_history_2.pkl')

# 从 pkl 文件加载数据
with open(file_path[0], 'rb') as file:
    loaded_datas = pickle.load(file)
results_2 = []
for load_idx, loaded_data in enumerate(loaded_datas):
    # 初始化列存储
    num_columns = len(loaded_data[0])
    columns = [[] for _ in range(num_columns)]
    for row in loaded_data:
        for col_idx, value in enumerate(row):
            columns[col_idx].append(value)
    result_2 = []
    for i in range(num_columns):
        # 数据过滤
        filtered = [x for x in columns[i] if x != 0 and x != float('inf')]
        # 去除连续重复值
        filtered_clean = []
        prev = None
        for x in filtered:
            if x != prev:
                filtered_clean.append(x)
                prev = x
        wait_agv_times = []

        # 遍历列表
        j = 0
        while j < len(filtered_clean):
            if filtered_clean[j] == -2:
                # 检查接下来的两个数
                if j + 2 < len(filtered_clean) and filtered_clean[j + 1] not in [-1, -2, -3] and filtered_clean[j + 2] not in [
                    -1, -2, -3]:
                    start_time = filtered_clean[j + 1]
                    end_time = filtered_clean[j + 2]
                    wait_agv_times.append((start_time, end_time))
                    j += 3  # 跳过已处理的 3 个元素
                elif j + 2 < len(filtered_clean) and filtered_clean[j + 1] not in [-1, -2, -3]:
                    start_time = filtered_clean[j + 1]
                    end_time = filtered_clean[j + 3]
                    wait_agv_times.append((start_time, end_time))
                    j += 4  # 跳过已处理的 4 个元素
            else:
                j += 1  # 继续检查下一个元素
        agv_wait = []
        for x in range(len(wait_agv_times)):
            agv_wait.append({
                'start': wait_agv_times[x][0], 'end': wait_agv_times[x][1]
            })
        result_2.append(agv_wait)
    results_2.append(result_2)

# 先找到等待生产区状态
file_path = glob.glob('Gantt_Chart/timeline_history_3.pkl')

# 从 pkl 文件加载数据
with open(file_path[0], 'rb') as file:
    loaded_datas = pickle.load(file)
results_3 = []
for load_idx, loaded_data in enumerate(loaded_datas):
    # 初始化列存储
    num_columns = len(loaded_data[0])
    columns = [[] for _ in range(num_columns)]
    for row in loaded_data:
        for col_idx, value in enumerate(row):
            columns[col_idx].append(value)

    result_3 = []
    for i in range(num_columns):
        # 数据过滤
        filtered = [x for x in columns[i] if x != 0]
        # 从后往前遍历列表，避免在删除元素时影响后面的元素
        i = 0
        while i < len(filtered) - 2:
            if filtered[i] == filtered[i+1] == filtered[i+2]:  # 如果存在三个连续的相同值
                del filtered[i+2]  # 删除第三个元素
            else:
                i += 1  # 如果没有连续三个相同元素，继续向前走
        # 如果列表长度为奇数，在第一位插入0
        if len(filtered) % 2 != 0:
            filtered.insert(0, 0)
        # 将列表每两个值分为一组，前一个为开始时间，后一个为结束时间
        wait_for_work = []
        wait_times = [(filtered[i], filtered[i + 1]) for i in range(0, len(filtered), 2)]

        agv_wait = []
        for x in range(len(wait_times)):
            wait_for_work.append({
                'start': wait_times[x][0], 'end': wait_times[x][1]
            })
        result_3.append(wait_for_work)
    results_3.append(result_3)
# 绘制甘特图
def plot_gantt_for_order(load_idx, results, results_1, results_2, results_3=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    # 自定义颜色映射（使用16进制颜色代码）
    color_map = {
        0: '#9bbf8a',  # 订单 1 使用橙色
        1: '#82afda',  # 订单 2 使用蓝色
        2: '#f79059',  # 订单 3 使用绿色
        3: '#e7dbd3',  # 订单 1 使用橙色
        4: '#c2bdde',  # 订单 2 使用蓝色
        5: '#8dcec8',  # 订单 3 使用绿色
        6: '#add3e2',  # 订单 1 使用橙色
        7: '#3480b8',  # 订单 2 使用蓝色
        8: '#fa8878',  # 订单 3 使用绿色
        9: '#ffbe7a',  # 订单 1 使用橙色
        10: '#8a6eaf',  # 浅薰衣草紫（补充紫色系）
        11: '#e69aaf',  # 玫瑰粉（增加暖色层次）
        12: '#a0a97f',  # 橄榄绿（大地色系过渡）
        13: '#5a9b8f',  # 深青瓷色（增加冷色深度）
        14: '#c8a2c8',  # 淡紫丁香（柔和中性色）
    }
    # ========== 图例样式优化 ==========
    # 创建专业图例句柄
    legend_elements = [
        mpatches.Patch(
            facecolor=color_map[6],
            edgecolor='black',
            label='生产区处理',
            linewidth=0.5
        ),
        mpatches.Patch(
            facecolor=color_map[2],
            edgecolor='black',
            label='AGV运输中',
            linewidth=0.5
        ),
        mpatches.Patch(
            facecolor=color_map[9],
            edgecolor='black',
            label='等待AGV',
            linewidth=0.5
        ),
        mpatches.Patch(
            facecolor='white',
            edgecolor='black',
            label='等待生产区处理',
            linewidth=0.5
        )
    ]
    # 为Orders绘制条形
    for order_idx, tasks in enumerate(results[load_idx]):
        for task in tasks:
            # 处理过程
            ax.barh(
                y=order_idx,
                width=task['end'] - task['start'],
                left=task['start'],
                height=0.4,
                color=color_map[6],
                edgecolor='black',
            )
            # ax.text(
            #     x=(task['start'] + task['end']) / 2,
            #     y=order_idx,
            #     s=f"生产区处理\n{task['end'] - task['start']:.1f}s",
            #     ha='center',
            #     va='center',
            #     color='black',
            #     fontsize=8
            # )
    for order_idx, tasks in enumerate(results_1[load_idx]):
        for task in tasks:
            # 运输过程
            ax.barh(
                y=order_idx,
                width=task['end'] - task['start'],
                left=task['start'],
                height=0.4,
                color=color_map[2],
                edgecolor='black',
            )
            # ax.text(
            #     x=(task['start'] + task['end']) / 2,
            #     y=order_idx,
            #     s=f"AGV运输\n{task['end'] - task['start']:.1f}s",
            #     ha='center',
            #     va='center',
            #     color='black',
            #     fontsize=8
            # )
    # 为Orders绘制条形
    for order_idx, tasks in enumerate(results_2[load_idx]):
        for task in tasks:
            # 等待小车过程
            ax.barh(
                y=order_idx,
                width=task['end'] - task['start'],
                left=task['start'],
                height=0.4,
                color=color_map[9],
                edgecolor='black',
            )
            # ax.text(
            #     x=(task['start'] + task['end']) / 2,
            #     y=order_idx,
            #     s=f"等待AGV小车\n{task['end'] - task['start']:.1f}s",
            #     ha='center',
            #     va='center',
            #     color='black',
            #     fontsize=8
            # )
    # for order_idx, tasks in enumerate(results_3[load_idx]):
    #     for task in tasks:
    #         # 等待生产区过程
    #         ax.barh(
    #             y=order_idx,
    #             width=task['end'] - task['start'],
    #             left=task['start'],
    #             height=0.4,
    #             color=color_map[3],
    #             edgecolor='black',
    #
    #         )
    # 设置坐标轴标签
    ax.set_xlim(left=0)  # 设置x轴的最小值为0
    ax.set_yticks(range(len(results_1[load_idx])))
    ax.set_yticklabels([f'Order {i+1}' for i in range(len(results_1[load_idx]))])
    ax.set_xlabel('Time')
    ax.set_title('订单处理甘特图')
    # 反向y轴
    ax.invert_yaxis()
    # 设置图例
    # ========== 图例配置优化 ==========
    ax.legend(
        handles=legend_elements,
        title='任务类型说明',
        title_fontsize=12,
        fontsize=10,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),  # 调整图例位置到右侧外部
        borderaxespad=0.5,
        framealpha=0.95,
        handlelength=2.5,
        handleheight=1.2,
        labelspacing=1.2
    )
    plt.tight_layout()
    save_path = f'results/result_{load_idx}/'
    os.makedirs(save_path, exist_ok=True)
    file_name = 'order_gantt.png'  # 文件名包含案例编号
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形避免内存泄漏

for load_idx in range(len(results)):
    plot_gantt_for_order(load_idx, results, results_1, results_2, results_3=None)

# 生产区甘特图
file_path = glob.glob('Gantt_Chart/timeline_history.pkl')
# 从 pkl 文件加载数据
with open(file_path[0], 'rb') as file:
    loaded_datas = pickle.load(file)
for load_idx, loaded_data in enumerate(loaded_datas):
    zones_path = glob.glob('Scatter_plot/0.pkl')
    # 从 pkl 文件加载数据
    with open(zones_path[0], 'rb') as file:
        zones_data = pickle.load(file)
        unit_distribution = zones_data['distributions_dict']
        # 生产区的有效数量列表
        production_zone_sizes = unit_distribution[load_idx]
    # 计算每个生产区的有效生产单元数
    valid_units = {area: sum(1 for value in values if value != 0) for area, values in production_zone_sizes.items()}

    # 仅保留元组
    filtered_data = [item for item in loaded_data if isinstance(item, tuple)]
    # 过滤掉元组，保留列表
    loaded_data = [row for row in loaded_data if not isinstance(row, tuple)]
    # 将矩阵转为 numpy 数组
    matrix_np = np.array(loaded_data)

    # 记录每两行的变化
    changes = []

    # 遍历每两行并记录变化的列
    for i in range(0, len(matrix_np) - 1, 2):
        start_row = matrix_np[i]  # 开始时间（当前行）
        end_row = matrix_np[i + 1]  # 结束时间（下一行）

        for j in range(len(start_row)):
            if start_row[j] != end_row[j]:  # 如果该列发生了变化
                changes.append((start_row[j], end_row[j], j))  # 记录变化的开始时间和结束时间
    # 合并两个列表
    combined_data = [(changes[i][0], changes[i][1], changes[i][2], filtered_data[i][0], filtered_data[i][1]) for i in range(len(filtered_data))]
    cope_with_orders = []
    results = []
    # 创建画布
    plt.figure(figsize=(12, 6))
    # 为每个订单分配一个颜色
    # 自定义颜色映射（使用16进制颜色代码）
    order_color_map = {
        0: '#9bbf8a',  # 订单 0 使用橙色
        1: '#82afda',  # 订单 1 使用蓝色
        2: '#f79059',  # 订单 2 使用绿色
        3: '#e7dbd3',  # 订单 3 使用橙色
        4: '#c2bdde',  # 订单 4 使用蓝色
        5: '#8dcec8',  # 订单 5 使用绿色
        6: '#add3e2',  # 订单 6 使用橙色
        7: '#3480b8',  # 订单 7 使用蓝色
        8: '#fa8878',  # 订单 8 使用绿色
        9: '#ffbe7a',  # 订单 9 使用橙色
        10: '#8a6eaf',  # 浅薰衣草紫（补充紫色系）
        11: '#e69aaf',  # 玫瑰粉（增加暖色层次）
        12: '#a0a97f',  # 橄榄绿（大地色系过渡）
        13: '#5a9b8f',  # 深青瓷色（增加冷色深度）
        14: '#c8a2c8',  # 淡紫丁香（柔和中性色）

        # 可以继续添加其他订单及其对应颜色
    }

    # 用来计算总的y位置
    current_position = 0

    # 绘制每个区的甘特图
    for zone, max_units in valid_units.items():
        zone_data = [d for d in combined_data if d[3] == zone]

        # 绘制每个单元的任务
        for unit in range(max_units):
            unit_tasks = [t for t in zone_data if t[4] == unit]

            for start, end, order_number,_, _ in unit_tasks:
                task_color = order_color_map[order_number]
                plt.barh(
                    y=current_position + unit,
                    width=end - start,
                    left=start,
                    height=0.8,
                    color=task_color,
                    edgecolor='black',
                    alpha=0.8,
                    label=f'订单 {order_number}' if unit == 0 else ''
                )
                # 添加任务标签
                plt.text(
                    x=(start + end) / 2,
                    y=current_position + unit,
                    s=f'订单{order_number+1}\n{end - start:.1f}s',
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=8
                )
                # # 添加任务标签
                # plt.text(
                #     x=start,
                #     y=current_position + unit,
                #     s=f'{start:.1f}s',
                #     ha='center',
                #     va='center',
                #     color='black',
                #     fontsize=8
                # )

        # 更新位置，以便下一个区域显示在下方
        current_position += max_units

    # 设置Y轴标签
    plt.yticks(range(current_position),
               [f'{zone} 单元 {unit+1}' for zone, max_units in valid_units.items() for unit in range(max_units)])

    # 添加图表和坐标轴标题
    plt.xlabel('时间')
    plt.ylabel('生产单元')
    plt.title('所有生产区的甘特图')

    # 创建自定义的图例
    legend_handles = [Line2D([0], [0], color=order_color_map[order_number], lw=4) for order_number in order_color_map]
    legend_labels = [f'订单 {order_number+1}' for order_number in order_color_map]
    # 显示图例
    plt.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    # 反向y轴
    plt.gca().invert_yaxis()
    # 显示图表
    plt.tight_layout()
    save_path = f'results/result_{load_idx}/'
    os.makedirs(save_path, exist_ok=True)
    file_name = 'zone_gantt.png'  # 文件名包含案例编号
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形避免内存泄漏
#
# # ------------------------
# # 按生产区划分的生产单元甘特图
# # # 为每个生产区单独绘图
# # for zone in valid_units.keys():
# #     # 创建新画布
# #     plt.figure(figsize=(12, 6))
# #
# #     # 获取该区配置
# #     max_units = valid_units[zone]
# #     zone_data = [d for d in combined_data if d[2] == zone]
# #
# #     # 创建y轴标签（即使单元未被使用）
# #     y_ticks = list(range(max_units))
# #     y_labels = [f'单元 {i}' for i in range(max_units)]
# #
# #     # 绘制每个单元的任务
# #     for unit in range(max_units):
# #         unit_tasks = [t for t in zone_data if t[3] == unit]
# #
# #         # 生成该单元专用颜色（保证颜色一致性）
# #         colors = generate_colors(len(unit_tasks))
# #
# #         # 绘制任务块
# #         for i, (start, end, _, _) in enumerate(unit_tasks):
# #             plt.barh(
# #                 y=unit,
# #                 width=end - start,
# #                 left=start,
# #                 height=0.6,
# #                 color=colors[i],
# #                 edgecolor='black',
# #                 alpha=0.8,
# #                 label=f'任务{i + 1}' if unit == 0 else ""  # 只在第一个单元显示图例
# #             )
# #
# #     # 设置坐标轴
# #     plt.yticks(y_ticks, y_labels)
# #     plt.ylim(-0.5, max_units - 0.5)  # 固定显示所有单元位置
# #     plt.xlabel('时间')
# #     plt.ylabel(f'{zone}')
# #
# #     # 动态调整X轴范围
# #     max_time = max([d[1] for d in combined_data]) * 1.05
# #     plt.xlim(0, max_time)
# #
# #     # 添加网格和标题
# #     plt.grid(axis='x', linestyle='--', alpha=0.7)
# #     plt.title(f'{zone}生产甘特图（共{max_units}个单元）')
# #
# #     # 显示图例（只显示一次）
# #     handles, labels = plt.gca().get_legend_handles_labels()
# #     if handles:
# #         plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
# #
# #     # 优化布局
# #     plt.tight_layout()
# #
# #     # 保存或显示
# #     plt.savefig(f'{zone}_gantt.png', dpi=300, bbox_inches='tight')
# #     plt.show()