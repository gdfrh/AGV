import matplotlib.pyplot as plt

# 示例数据（多个生产单元）
tasks = [
    {"Unit": "A1", "AGV": "AGV1", "Start": 10, "Finish": 20},
    {"Unit": "A1", "AGV": "AGV2", "Start": 20, "Finish": 35},
    {"Unit": "A2", "AGV": "AGV1", "Start": 12.5, "Finish": 25},
    {"Unit": "A2", "AGV": "AGV3", "Start": 30, "Finish": 45},
    {"Unit": "A3", "AGV": "AGV2", "Start": 5, "Finish": 15},
]

# 提取唯一的生产单元并按顺序分配 Y 轴位置
units = sorted(list({task["Unit"] for task in tasks}))  # 按字母排序，如 ["A1", "A2", "A3"]
print(units)
y_positions = {unit: idx for idx, unit in enumerate(units)}

# 创建图表
fig, ax = plt.subplots()

# 绘制每个任务条
for task in tasks:
    # 计算任务参数
    y = y_positions[task["Unit"]]  # 根据生产单元分配 Y 轴位置
    start = task["Start"]
    duration = task["Finish"] - start
    color = "red" if task["AGV"] == "AGV1" else "skyblue"

    # 绘制水平条形图
    ax.barh(y=y, width=duration, left=start, height=0.5,
            color=color, edgecolor="black", alpha=0.8)

    # 添加AGV名称文本（居中显示）
    text_x = start + duration / 2
    ax.text(text_x, y, task["AGV"],
            ha='center', va='center', color='black', fontweight='bold')

# 设置Y轴标签
ax.set_yticks([y_positions[unit] for unit in units])  # Y 轴刻度位置
ax.set_yticklabels(units)  # Y 轴标签文本（显示在左侧）
ax.set_xlabel("Time")
ax.set_ylabel("Production Units")
ax.set_title("Multi-Unit AGV Schedule")

# 设置网格和布局
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
