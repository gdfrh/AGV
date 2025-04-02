import matplotlib.pyplot as plt
import pickle
import glob
file_path = glob.glob(' Gantt_Chart/agv_timeline_history.pkl')
# 从 pkl 文件加载矩阵
with open(file_path[0], 'rb') as file:
    loaded_data = pickle.load(file)
    # for i in range(len(loaded_data)):
    #     print(loaded_data[i])
# 获取矩阵的列数
num_columns = len(loaded_data[0])

# 按列提取数据
columns = [[] for _ in range(num_columns)]  # 初始化列的列表
filtered_column = [[] for _ in range(num_columns)]  # 初始化列的列表
for row in loaded_data:
    for col_idx, value in enumerate(row):
        columns[col_idx].append(value)

# 输出按列提取的数据
for i, col in enumerate(columns):
    # 过滤操作
    filtered_column_new = []
    previous_value = None  # 用来跟踪前一个元素
    filtered_column[i] = [x for x in col if x != -1 and (not isinstance(x, tuple) or x[0] is not None)]
    for x in filtered_column[i]:
        # 检查是否符合过滤条件
        if x != -1 and (not isinstance(x, tuple) or x[0] is not None):
            # 如果当前元素与前一个元素相同，则跳过
            if x != previous_value:
                filtered_column_new.append(x)
                previous_value = x  # 更新前一个元素
    # 结果存储列表
    result = []

    # 遍历列表，每三个元素一组
    for j in range(0, len(filtered_column_new) - 2, 3):
        if filtered_column_new:
            # 获取小车运输订单的开始和结束时间

            transport_start = filtered_column_new[j]
            transport_end = filtered_column_new[j + 1][0]  # 第二个元素的第一个值是小车返回生产区的开始时间

            # 获取小车返回生产区的开始和结束时间
            return_start = filtered_column_new[j + 1][0]  # 第二个元素的第一个值
            return_end = filtered_column_new[j + 2][0]  # 第三个元素的第一个值

            # 存储结果
            result.append({
                '运输开始时间': transport_start,
                '运输结束时间': transport_end,
                '返回生产区开始时间': return_start,
                '返回生产区结束时间': return_end
            })

    # 打印结果
    for entry in result:
        print(entry)
# def extract_values_with_first_zero(matrix, column_index):
#     values = []
#     first_zero_added = False
#
#     for row in matrix:
#         value = row[column_index]
#         if value != 0 or not first_zero_added:
#             values.append(value)
#             if value == 0:
#                 first_zero_added = True
#
#     return values
# # length = len(loaded_matrix[0])
# for i in range(len(loaded_matrix)):
#     print(loaded_matrix[i])
# # 调用函数并提取第二列的所有非零值
# tasks = []
# # column_index = 0  # 第二列
# # nonzero_values = extract_values_with_first_zero(loaded_matrix, column_index)
# for i in range(len(loaded_matrix[0])):
#     nonzero_values = extract_values_with_first_zero(loaded_matrix, i)
#     for j in range(len(nonzero_values)):
#         if nonzero_values[j] == -1:
#             task ={"Order": i, "AGV": f"AGV{i}", "Start": nonzero_values[j + 1], "Finish": nonzero_values[j + 2]}
#             tasks.append(task)
#
# # 提取唯一的生产单元并按顺序分配 Y 轴位置
# Orders = sorted(list({task["Order"] for task in tasks}))  # 按字母排序，如 ["A1", "A2", "A3"]
# print(Orders)
# y_positions = {unit: idx for idx, unit in enumerate(Orders)}
#
# # 创建图表
# fig, ax = plt.subplots()
#
# # 绘制每个任务条
# for task in tasks:
#     # 计算任务参数
#     y = y_positions[task["Order"]]  # 根据生产单元分配 Y 轴位置
#     start = task["Start"]
#     duration = task["Finish"] - start
#     color = "red" if task["AGV"] == "AGV1" else "skyblue"
#
#     # 绘制水平条形图
#     ax.barh(y=y, width=duration, left=start, height=0.5,
#             color=color, edgecolor="black", alpha=0.8)
#
#     # # 添加AGV名称文本（居中显示）
#     # text_x = start + duration / 2
#     # ax.text(text_x, y, task["AGV"],
#     #         ha='center', va='center', color='black', fontweight='bold')
#
# # 设置Y轴标签
# ax.set_yticks([y_positions[unit] for unit in Orders])  # Y 轴刻度位置
# ax.set_yticklabels(Orders)  # Y 轴标签文本（显示在左侧）
# ax.set_xlabel("Time")
# ax.set_ylabel("Production Units")
# ax.set_title("Multi-Unit AGV Schedule")
#
# # 设置网格和布局
# plt.grid(axis="x", linestyle="--", alpha=0.7)
# plt.tight_layout()
# plt.show()
#
#
#
#
