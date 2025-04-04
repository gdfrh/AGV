import matplotlib.pyplot as plt
import pickle
import glob
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib import rcParams
from Config import *

# # 设置 matplotlib 使用中文字体
# rcParams['font.family'] = 'SimHei'  # 使用黑体（SimHei）字体，或根据需要选择其他字体
#
# # 如果需要支持更好的中文显示，可能还需要设置 'font.sans-serif'
# rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei' 等
#
#
# file_path = glob.glob('Scatter_plot/0.pkl')
# # 从 pkl 文件加载数据
# with open(file_path[0], 'rb') as file:
#     loaded_data = pickle.load(file)
#     agv_distribution = loaded_data['agv_distribution']
#     # 生产区的小车数量列表
#     production_zone_sizes = agv_distribution[0]
#
# # 修正文件路径中的空格
# file_path = glob.glob('Gantt_Chart/agv_timeline_history.pkl')
#
# # 从 pkl 文件加载数据
# with open(file_path[0], 'rb') as file:
#     loaded_data = pickle.load(file)
#
# # 初始化列存储
# num_columns = len(loaded_data[0])
# columns = [[] for _ in range(num_columns)]
# for row in loaded_data:
#     for col_idx, value in enumerate(row):
#         columns[col_idx].append(value)
# results = []
# for i in range(num_columns):
#     # 数据过滤
#     filtered = [x for x in columns[i] if x != -1 and (not isinstance(x, tuple) or x[0] is not None)]
#     # 去除连续重复值
#     filtered_clean = []
#     prev = None
#     for x in filtered:
#         if x != prev:
#             filtered_clean.append(x)
#             prev = x
#
#     # 分组处理（每3个元素一组）
#     car_tasks = []
#     for j in range(0, len(filtered_clean) - 2, 3):
#         transport_start = filtered_clean[j]
#         return_start = filtered_clean[j + 1][0]
#         return_end = filtered_clean[j + 2][0]
#
#         # 计算运输结束时间为返回开始时间
#         transport_end_time = return_start
#
#         car_tasks.append({
#             'transport': (transport_start, transport_end_time),
#             'return': (return_start, return_end)
#         })
#
#     results.append(car_tasks)
# # 逐个生产区绘制图形
# start_idx = 0  # 用于跟踪每个生产区的小车起始位置
# for zone_idx, num_agvs in enumerate(production_zone_sizes):
#     fig, ax = plt.subplots(figsize=(10, 5))
#
#     ax.set_title(f'{work_name[zone_idx]} AGV Schedule')
#
#     # 在每个生产区绘制小车的任务
#     for agv_idx in range(num_agvs):
#         agv_tasks = results[start_idx + agv_idx]  # 获取当前小车的任务
#
#         for task in agv_tasks:
#             # 运输过程
#             ax.barh(
#                 y=agv_idx,
#                 width=task['transport'][1] - task['transport'][0],
#                 left=task['transport'][0],
#                 height=0.4,
#                 color='skyblue',
#                 edgecolor='black',
#                 label='Transport' if agv_idx == 0 else ""
#             )
#
#             # 返回过程
#             ax.barh(
#                 y=agv_idx,
#                 width=task['return'][1] - task['return'][0],
#                 left=task['return'][0],
#                 height=0.4,
#                 color='orange',
#                 edgecolor='black',
#                 label='Return' if agv_idx == 0 else ""
#             )
#
#     # 设置坐标轴标签
#     ax.set_yticks(range(num_agvs))  # 每个小车的索引
#     ax.set_yticklabels([f'AGV {i + 1}' for i in range(num_agvs)])
#     ax.set_xlabel('Time')
#     ax.set_ylabel('AGV')
#
#     start_idx += num_agvs  # 更新起始小车索引
#     # 设置时间轴从0开始
#     ax.set_xlim(left=0)  # 将时间轴的左端设置为0
#
#     # 显示每个生产区的甘特图
#     plt.tight_layout()
#     plt.show()

# 订单甘特图
# 找到处理订单部分
file_path = glob.glob('Gantt_Chart/timeline_history.pkl')

# 从 pkl 文件加载数据
with open(file_path[0], 'rb') as file:
    loaded_data = pickle.load(file)

# 初始化列存储
num_columns = len(loaded_data[0])
columns = [[] for _ in range(num_columns)]
for row in loaded_data:
    for col_idx, value in enumerate(row):
        columns[col_idx].append(value)
results = []
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
    results.append(cope_with_orders)

# 找到小车运输状态
file_path = glob.glob('Gantt_Chart/timeline_history_1.pkl')

# 从 pkl 文件加载数据
with open(file_path[0], 'rb') as file:
    loaded_data = pickle.load(file)
# for i in range(len(loaded_data)):
#     print(loaded_data[i])
# 初始化列存储
num_columns = len(loaded_data[0])
columns = [[] for _ in range(num_columns)]
for row in loaded_data:
    for col_idx, value in enumerate(row):
        columns[col_idx].append(value)
results_1 = []
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
    results_1.append(agv_transport)

# 先找到等待小车状态
file_path = glob.glob('Gantt_Chart/timeline_history_2.pkl')

# 从 pkl 文件加载数据
with open(file_path[0], 'rb') as file:
    loaded_data = pickle.load(file)
# for i in range(len(loaded_data)):
#     print(loaded_data[i])
# 初始化列存储
num_columns = len(loaded_data[0])
columns = [[] for _ in range(num_columns)]
for row in loaded_data:
    for col_idx, value in enumerate(row):
        columns[col_idx].append(value)

results_2 = []
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
    results_2.append(agv_wait)

# 先找到等待生产区状态
file_path = glob.glob('Gantt_Chart/timeline_history_3.pkl')

# 从 pkl 文件加载数据
with open(file_path[0], 'rb') as file:
    loaded_data = pickle.load(file)
# 初始化列存储
num_columns = len(loaded_data[0])
columns = [[] for _ in range(num_columns)]
for row in loaded_data:
    for col_idx, value in enumerate(row):
        columns[col_idx].append(value)


results_3 = []
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
    results_3.append(wait_for_work)
# 绘制甘特图
fig, ax = plt.subplots(figsize=(12, 6))
# 创建代理条形图对象用于图例（只有一条条形）
working_by_robot_arm = plt.Rectangle((0, 0), 1, 1, fc='pink', edgecolor='black')
transporting_by_agvs = plt.Rectangle((0, 0), 1, 1, fc='skyblue', edgecolor='black')
waiting_for_agvs = plt.Rectangle((0, 0), 1, 1, fc='green', edgecolor='black')
waiting_for_working = plt.Rectangle((0, 0), 1, 1, fc='yellow', edgecolor='black')
# 为Orders绘制条形
for order_idx, tasks in enumerate(results):
    for task in tasks:
        # 处理过程
        ax.barh(
            y=order_idx,
            width=task['end'] - task['start'],
            left=task['start'],
            height=0.4,
            color='pink',
            edgecolor='black',

        )
for order_idx, tasks in enumerate(results_1):
    for task in tasks:
        # 运输过程
        ax.barh(
            y=order_idx,
            width=task['end'] - task['start'],
            left=task['start'],
            height=0.4,
            color='skyblue',
            edgecolor='black',

        )
# 为Orders绘制条形
for order_idx, tasks in enumerate(results_2):
    for task in tasks:
        # 等待小车过程
        ax.barh(
            y=order_idx,
            width=task['end'] - task['start'],
            left=task['start'],
            height=0.4,
            color='green',
            edgecolor='black',

        )
for order_idx, tasks in enumerate(results_3):
    for task in tasks:
        # 等待生产区过程
        ax.barh(
            y=order_idx,
            width=task['end'] - task['start'],
            left=task['start'],
            height=0.4,
            color='yellow',
            edgecolor='black',

        )
# 设置坐标轴标签
ax.set_xlim(left=0)  # 设置x轴的最小值为0
ax.set_yticks(range(len(results_1)))
ax.set_yticklabels([f'Order {i}' for i in range(len(results_1))])
ax.set_xlabel('Time')
ax.set_title('Orders transport Gantt Chart')
# 反向y轴
ax.invert_yaxis()
# 设置图例
ax.legend([working_by_robot_arm, transporting_by_agvs, waiting_for_agvs,waiting_for_working],
          ['Working by robot_arm', 'Transporting by Agvs', 'Waiting for Agvs','Waiting for Working'], loc='upper right')
plt.tight_layout()
plt.show()

