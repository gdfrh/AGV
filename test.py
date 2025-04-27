# # # # # # # # import pickle
# # # # # # # #
# # # # # # # # # 假设这些是您要保存的数据
# # # # # # # # energy_pic = [1, 2, 3, 4]
# # # # # # # # time_pic = [4, 3, 2, 1]
# # # # # # # # agv_distributions = [[1, 0], [2, 1], [1, 1], [0, 0]]
# # # # # # # # order_distributions = [[1, 2], [2, 3], [3, 4], [4, 5]]
# # # # # # # # distributions_dicts = [{'zone1': [1, 2]}, {'zone2': [2, 3]}]
# # # # # # # #
# # # # # # # # # 将数据存储到文件
# # # # # # # # data = {
# # # # # # # #     'energy': energy_pic,
# # # # # # # #     'time': time_pic,
# # # # # # # #     'agv_distribution': agv_distributions,
# # # # # # # #     'order': order_distributions,
# # # # # # # #     'distributions_dict': distributions_dicts
# # # # # # # # }
# # # # # # # #
# # # # # # # # with open('data.pkl', 'wb') as file:
# # # # # # # #     pickle.dump(data, file)
# # # # # # # # import random
# # # # # # # #
# # # # # # # # # 示例数据：总数和各生产单元机器臂的初始分布
# # # # # # # # total_machines = 25  # 假设总机器臂数量为25
# # # # # # # # parts = [5, 5, 5, 5, 5]  # 假设有5个生产单元，每个生产单元初始分配的机器臂数量
# # # # # # # #
# # # # # # # # # 确保总数量不超过总机器臂数量
# # # # # # # # current_sum = sum(expanded_list)
# # # # # # # # remaining_machines = total_machines - current_sum
# # # # # # # #
# # # # # # # # # 将剩余的机器臂数量分配到已分配的机器臂数量中
# # # # # # # # for i in range(remaining_machines):
# # # # # # # #     # 随机选择一个生产单元，增加机器臂
# # # # # # # #     random_index = random.choice(range(len(parts)))
# # # # # # # #     expanded_list[random_index] += 1
# # # # # # # # import random
# # # # # # # #
# # # # # # # # list = [0,0,0]
# # # # # # # # print(len(list))
# # # # # # # # print(random.random(),random.random(),random.random())
# # # # # # # # import random
# # # # # # # # zone_units =[0,1,2,3,4,5]
# # # # # # # # unit_to_remove = random.choice(zone_units)
# # # # # # # # print(unit_to_remove)
# # # # # # # # zone_units.remove(unit_to_remove)
# # # # # # # # print(zone_units)
# # # # # # # # zone_units =[0,1,2,3,4,5]
# # # # # # # # zone_units.append(6)
# # # # # # # # zone_units = sorted(zone_units, reverse=True)
# # # # # # # # print(zone_units)
# # # # # # # # expanded_list = [0] + zone_units + [1]
# # # # # # # # expanded_list = sorted(expanded_list, reverse=True)
# # # # # # # # print(expanded_list)
# # # # # # # # arm_distributions={'a':1,'b':2,'c':3,'d':4,'e':5}
# # # # # # # # for zone in arm_distributions:
# # # # # # # #     print(zone)
# # # # # # # # 定义一个新字典
# # # # # # # import random
# # # # # # # from Config import *
# # # # # # # arm_distributions = {}
# # # # # # # for zone, unit_count in zip(work_name, [3,3,3,3,3,3,3]):
# # # # # # #     arm_distributions[zone] = [0] * unit_count  # 为每个生产单元初始化机器数为 0
# # # # # # #     # 1. 根据需求给每个生产区分配机器
# # # # # # # remaining_machines = total_machines  # 总机器数量
# # # # # # # for zone, min_machines in zone_requirements:  # zone_requirements 中保存每个生产区需要的最小机器数量
# # # # # # #     # 获取该生产区的单元数
# # # # # # #     units = len(arm_distributions[zone])
# # # # # # #     obj_units = random.randint(0, units - 1)
# # # # # # #     # 先给每个生产单元分配最低机器臂数量
# # # # # # #     arm_distributions[zone][obj_units] = min_machines
# # # # # # #     remaining_machines -= min_machines
# # # # # # # # 2. 随机分配机器
# # # # # # # while remaining_machines > 0:
# # # # # # #     zone = random.choice(work_name)  # 随机选择一个生产区
# # # # # # #     unit_index = random.randint(0, len(arm_distributions[zone]) - 1)  # 随机选择一个生产单元
# # # # # # #     # 分配每个生产区需要的机器数，而不是逐个机器分配
# # # # # # #     """如果机器臂数量为0，则分配最低要求数"""
# # # # # # #     if arm_distributions[zone][unit_index] == 0:
# # # # # # #         min_required_machines = zone_requirements[work_name.index(zone)][1]  # 获取该生产区的最小机器需求
# # # # # # #         if remaining_machines >= min_required_machines:
# # # # # # #             arm_distributions[zone][unit_index] = min_required_machines
# # # # # # #             remaining_machines -= min_required_machines
# # # # # # #         """如果已分配机器臂，就多分配1个"""
# # # # # # #     elif arm_distributions[zone][unit_index] != 0:
# # # # # # #         arm_distributions[zone][unit_index] += 1
# # # # # # #         remaining_machines -= 1
# # # # # # # # 3. 对每个生产区的生产单元按机器数量从大到小排序
# # # # # # # for zone in arm_distributions:
# # # # # # #     # 排序每个生产区的单元，按机器数量从大到小
# # # # # # #     arm_distributions[zone] = sorted(arm_distributions[zone], reverse=True)
# # # # # # # machine_count_list = []  # 列表
# # # # # # # for zone, units in arm_distributions.items():
# # # # # # #     # 将每个生产区的机器数存储到字典中
# # # # # # #     machine_count_list += units  # 每个单元的机器数
# # # # # # # merged_str = ''.join(map(str, machine_count_list))
# # # # # # # # 将连接后的字符串转换为整数
# # # # # # # merged_number = int(merged_str)
# # # # # # #
# # # # # # # print(merged_number)
# # # # # # import matplotlib.pyplot as plt
# # # # # # import matplotlib.colors as mcolors
# # # # # # import matplotlib.dates as mdates
# # # # # # from datetime import datetime, timedelta
# # # # # #
# # # # # # # 假设有如下数据：每个生产区的状态列表，包含开始时间和结束时间
# # # # # # # 例如：'work'代表工作，'idle'代表空闲，'wait'代表等待
# # # # # # work_status_data = {
# # # # # #     'Factory1': [
# # # # # #         {'status': 'work', 'start': '2025-03-29 00:00:00', 'end': '2025-03-29 02:00:00'},
# # # # # #         {'status': 'idle', 'start': '2025-03-29 02:00:00', 'end': '2025-03-29 04:00:00'},
# # # # # #         {'status': 'wait', 'start': '2025-03-29 04:00:00', 'end': '2025-03-29 06:00:00'},
# # # # # #         {'status': 'work', 'start': '2025-03-29 06:00:00', 'end': '2025-03-29 08:00:00'},
# # # # # #     ],
# # # # # #     'Factory2': [
# # # # # #         {'status': 'wait', 'start': '2025-03-29 00:00:00', 'end': '2025-03-29 01:30:00'},
# # # # # #         {'status': 'work', 'start': '2025-03-29 01:30:00', 'end': '2025-03-29 03:00:00'},
# # # # # #         {'status': 'idle', 'start': '2025-03-29 03:00:00', 'end': '2025-03-29 05:00:00'},
# # # # # #         {'status': 'work', 'start': '2025-03-29 05:00:00', 'end': '2025-03-29 07:00:00'},
# # # # # #     ]
# # # # # # }
# # # # # #
# # # # # # # 将时间字符串转换为时间对象
# # # # # # def parse_time(time_str):
# # # # # #     return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
# # # # # #
# # # # # # # 创建一个颜色映射：可以用不同的颜色表示不同状态
# # # # # # status_colors = {
# # # # # #     'work': mcolors.CSS4_COLORS['green'],  # 绿色表示工作
# # # # # #     'idle': mcolors.CSS4_COLORS['gray'],   # 灰色表示空闲
# # # # # #     'wait': mcolors.CSS4_COLORS['blue'],   # 蓝色表示等待
# # # # # # }
# # # # # #
# # # # # # # 创建甘特图
# # # # # # fig, ax = plt.subplots(figsize=(10, 6))
# # # # # #
# # # # # # # 设置y轴：生产区
# # # # # # y_labels = list(work_status_data.keys())
# # # # # # ax.set_yticks(range(len(y_labels)))
# # # # # # ax.set_yticklabels(y_labels)
# # # # # #
# # # # # # # 设置x轴时间范围
# # # # # # start_time = min(parse_time(status['start']) for zone in work_status_data.values() for status in zone)
# # # # # # end_time = max(parse_time(status['end']) for zone in work_status_data.values() for status in zone)
# # # # # # ax.set_xlim([start_time, end_time])
# # # # # #
# # # # # # # 格式化x轴的时间
# # # # # # ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
# # # # # # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H-%M'))
# # # # # #
# # # # # # # 绘制每个生产区的工作状态
# # # # # # for i, (zone, statuses) in enumerate(work_status_data.items()):
# # # # # #     for status in statuses:
# # # # # #         start = parse_time(status['start'])
# # # # # #         end = parse_time(status['end'])
# # # # # #         ax.barh(zone, end - start, left=start, color=status_colors[status['status']], edgecolor='black')
# # # # # #
# # # # # # # 添加标题和标签
# # # # # # ax.set_xlabel('Time')
# # # # # # ax.set_ylabel('Factory')
# # # # # # ax.set_title('Factory Work Status Gantt Chart')
# # # # # #
# # # # # # # 自动旋转日期标签
# # # # # # plt.xticks(rotation=45)
# # # # # #
# # # # # # # 显示图表
# # # # # # plt.tight_layout()
# # # # # # plt.show()
# # # # # timeline_history = []
# # # # # timeline_one = [1,2,3]
# # # # # timeline_one[0] = 5
# # # # # timeline_history.append(timeline_one)
# # # # # timeline_one = [2,3,4]
# # # # # print(timeline_history)
# # # # # timeline_history = []
# # # # # timeline_one = [1, 2, 3]
# # # # # timeline_one[0] = 5
# # # # # timeline_history.append(timeline_one)
# # # # # print(timeline_history)  # 输出: [[5, 2, 3]]，timeline_one 的修改影响到了 timeline_history 中的内容
# # # # #
# # # # # # 更改 timeline_one
# # # # # timeline_one = [2, 3, 4]
# # # # # timeline_history.append(timeline_one)
# # # # # print(timeline_history)  # 输出: [[5, 2, 3], [2, 3, 4]]
# # # # # 过滤操作
# # # # # a = []
# # # # #
# # # # # previous_value = None  # 用来跟踪前一个元素
# # # # # col = [1,2,2,3,3,4]
# # # # # for x in col:
# # # # #     # 检查是否符合过滤条件
# # # # #     if x != -1 and (not isinstance(x, tuple) or x[0] is not None):
# # # # #         # 如果当前元素与前一个元素相同，则跳过
# # # # #         if x != previous_value:
# # # # #             a.append(x)
# # # # #             previous_value = x  # 更新前一个元素
# # # # # print(a)
# # # # import numpy as np
# # # #
# # # #
# # # # # 生成参考点函数
# # # # def generate_reference_points(num_obj, divisions):
# # # #     """
# # # #     生成参考点
# # # #     num_obj: 目标数目
# # # #     divisions: 每个目标的分割数目（分布密度）
# # # #     """
# # # #     # 生成一个等间距的参考点网格
# # # #     reference_points = []
# # # #     step_size = 1.0 / (divisions - 1)
# # # #
# # # #     # 使用多维网格方法生成参考点
# # # #     grid = np.array(np.meshgrid(*[np.linspace(0, 1, divisions)] * num_obj))
# # # #     grid = grid.reshape(num_obj, -1).T
# # # #
# # # #     # 将参考点按列归一化到 [0, 1] 区间
# # # #     reference_points = grid
# # # #
# # # #     return reference_points
# # # #
# # # #
# # # # # 设置目标数目和分布密度
# # # # num_obj = 2  # 目标数目
# # # # divisions = 10  # 分布密度，表示每个维度分为4个区间
# # # #
# # # # # 生成参考点
# # # # ref_points = generate_reference_points(num_obj, divisions)
# # # #
# # # # # 初始化理想点和截距点
# # # # ideal_point = np.full(num_obj, np.inf)
# # # # nadir_point = np.full(num_obj, -np.inf)
# # # #
# # # # print("生成的参考点：")
# # # # print(ref_points)
# # # # import os
# # # #
# # # #
# # # # def get_next_filename(directory, base_name):
# # # #     # 获取目录下所有的 .pkl 文件
# # # #     files = [f for f in os.listdir(directory) if f.endswith('.pkl') and f.startswith(base_name)]
# # # #
# # # #     # 提取所有的编号
# # # #     numbers = []
# # # #     for file in files:
# # # #         try:
# # # #             # 假设文件名格式为 "base_name_index.pkl"
# # # #             parts = file[len(base_name) + 1:].replace('.pkl', '')  # 取出索引部分
# # # #             numbers.append(int(parts))
# # # #         except ValueError:
# # # #             continue
# # # #
# # # #     # 如果没有找到文件，则从 0 开始
# # # #     next_number = max(numbers, default=-1) + 1
# # # #     return os.path.join(directory, f'{base_name}_{next_number}.pkl')
# # # #
# # # #
# # # # # 示例
# # # # directory = 'Pareto_Crowding_Distance'  # 文件夹路径
# # # # base_name = '1'  # 基础文件名部分
# # # # new_file_path = get_next_filename(directory, base_name)
# # # # print(new_file_path)
# # # # file_path = os.path.join('Pareto_Crowding_Distance', f'0_.pkl')
# # # # print(file_path)
# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # #
# # # # 示例数据生成（替换为你的实际数据）
# # # iterations = np.arange(0, 501, 10)  # 迭代次数0-500
# # # makespan = 180 - 0.15 * iterations + np.random.normal(0, 2, len(iterations))  # 最大完工时间模拟数据
# # # energy = 90 + 0.1 * iterations + np.random.normal(0, 1, len(iterations))  # 能耗模拟数据
# # #
# # # # 创建画布和子图
# # # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# # # plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示设置
# # # plt.rcParams['axes.unicode_minus'] = False
# # #
# # # # 绘制最大完工时间子图
# # # ax1.plot(iterations, makespan,
# # #         color='#2E75B6',
# # #         linewidth=2,
# # #         label='IMOSSA')
# # # ax1.set_ylabel('最大完工时间/min', fontsize=12)
# # # ax1.grid(True, linestyle='--', alpha=0.7)
# # # ax1.set_title('IMOSSA求解案例完工时间和能耗进化轨迹', fontsize=14, pad=20)
# # # ax1.legend(loc='upper right')
# # #
# # # # 绘制能耗子图
# # # ax2.plot(iterations, energy,
# # #         color='#C00000',
# # #         linewidth=2)
# # # ax2.set_xlabel('迭代次数', fontsize=12)
# # # ax2.set_ylabel('机器与AGV综合能耗/kWh', fontsize=12)
# # # ax2.grid(True, linestyle='--', alpha=0.7)
# # #
# # # # 设置公共参数
# # # for ax in [ax1, ax2]:
# # #     ax.set_xlim(0, 500)
# # #     ax.tick_params(axis='both', which='major', labelsize=10)
# # #
# # # ax1.set_ylim(80, 180)  # 根据实际数据调整
# # # ax2.set_ylim(80, 100)  # 根据实际数据调整
# # #
# # # plt.tight_layout()
# # # plt.show()
# # #
# # import os
# # import re
# # import pickle
# # data={}
# # compare=0
# # # 1. 定义要创建的文件夹名称
# # folder_name = "Scatter_plot"  # 可以修改为任意文件夹名称/路径
# # # 2. 创建文件夹（如果不存在）
# # os.makedirs(folder_name, exist_ok=True)
# #
# # # 获取当前所有符合 {compare}_数字.pkl 格式的文件
# # existing_files = os.listdir(folder_name)
# # compare_str = str(compare)
# # pattern = re.compile(rf'^{re.escape(compare_str)}_(\d+)\.pkl$')  # 使用正则匹配带数字编号的文件
# #
# # # 找到最大编号
# # max_idx = -1
# # for filename in existing_files:
# #     match = pattern.match(filename)
# #     if match:
# #         current_idx = int(match.group(1))
# #         max_idx = max(max_idx, current_idx)
# #
# # # 计算新编号（最大编号+1）
# # idx = max_idx + 1
# #
# # # 3. 构造完整文件路径
# # file_path = os.path.join(folder_name, f'{compare_str}_{idx}.pkl')
# #
# # # 4. 保存文件到新文件夹
# # with open(file_path, 'wb') as file:
# #     pickle.dump(data, file)
# # 绘制迭代图（横向布局）
# import pickle
# from Config import *
# import plotly.express as px
# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime
# import glob
# import matplotlib.pyplot as plt
# from deap import base, creator, tools
# file_path = glob.glob('iterations/iterations.pkl')
# with open(file_path[0], 'rb') as file:
#     loaded_data = pickle.load(file)
#
# for case_idx, case_data in enumerate(loaded_data):
#     if case_idx == 4:
#         # 遍历列表，如果小于100就改成100
#         makespan = sorted(case_data[0], reverse=True)
#         print(makespan)
#         data1 = [417.5 if x < 417.5 else x for x in case_data[0]]
#         data2 = case_data[1]
#         data = (data1, data2)
#         # print(data)
#         # print(case_data)
#         loaded_data[case_idx] = data
# # 保存修改后的数据回文件
# with open(file_path[0], 'wb') as file:
#     pickle.dump(loaded_data, file)