import matplotlib.pyplot as plt
import pickle

# 从 pkl 文件加载矩阵
with open('timeline_history.pkl', 'rb') as file:
    loaded_matrix = pickle.load(file)
def extract_values_with_first_zero(matrix, column_index):
    values = []
    first_zero_added = False

    for row in matrix:
        value = row[column_index]
        if value != 0 or not first_zero_added:
            values.append(value)
            if value == 0:
                first_zero_added = True

    return values
# length = len(loaded_matrix[0])
for i in range(len(loaded_matrix)):
    print(loaded_matrix[i])
# 调用函数并提取第二列的所有非零值
tasks = []
# column_index = 0  # 第二列
# nonzero_values = extract_values_with_first_zero(loaded_matrix, column_index)
for i in range(len(loaded_matrix[0])):
    nonzero_values = extract_values_with_first_zero(loaded_matrix, i)
    for j in range(len(nonzero_values)):
        if nonzero_values[j] == -1:
            task ={"Order": i, "AGV": f"AGV{i}", "Start": nonzero_values[j + 1], "Finish": nonzero_values[j + 2]}
            tasks.append(task)

# 提取唯一的生产单元并按顺序分配 Y 轴位置
Orders = sorted(list({task["Order"] for task in tasks}))  # 按字母排序，如 ["A1", "A2", "A3"]
print(Orders)
y_positions = {unit: idx for idx, unit in enumerate(Orders)}

# 创建图表
fig, ax = plt.subplots()

# 绘制每个任务条
for task in tasks:
    # 计算任务参数
    y = y_positions[task["Order"]]  # 根据生产单元分配 Y 轴位置
    start = task["Start"]
    duration = task["Finish"] - start
    color = "red" if task["AGV"] == "AGV1" else "skyblue"

    # 绘制水平条形图
    ax.barh(y=y, width=duration, left=start, height=0.5,
            color=color, edgecolor="black", alpha=0.8)

    # # 添加AGV名称文本（居中显示）
    # text_x = start + duration / 2
    # ax.text(text_x, y, task["AGV"],
    #         ha='center', va='center', color='black', fontweight='bold')

# 设置Y轴标签
ax.set_yticks([y_positions[unit] for unit in Orders])  # Y 轴刻度位置
ax.set_yticklabels(Orders)  # Y 轴标签文本（显示在左侧）
ax.set_xlabel("Time")
ax.set_ylabel("Production Units")
ax.set_title("Multi-Unit AGV Schedule")

# 设置网格和布局
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()





