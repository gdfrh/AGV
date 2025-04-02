import pickle
import plotly.express as px
import pandas as pd
import glob
import os

# 替换 'folder_name' 为你目标文件夹的路径
file_paths = glob.glob('Bar_plot/*.pkl')

# 用于存储所有的数据
combined_data = {
    'time': [],
    'group': [],  # 用于存储每个数据来源的 group 信息
}
# 遍历所有文件
for idx, file_path in enumerate(file_paths):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    # 获取文件中的数据
    time_pic = loaded_data['time']

    # 为每个文件的数据添加一个group字段
    # 从文件路径中提取文件名作为 group_label
    group_label = os.path.basename(file_path).replace('.pkl', '')  # 去掉 .pkl 后缀，作为 group 标签  # 给每个文件添加一个独特的标识符
    combined_data['time'] += time_pic
    combined_data['group'] += [group_label] * len(time_pic)  # 对应的每个数据点分配相同的 group 标签

# 将数据转换为 DataFrame
df = pd.DataFrame(combined_data)

# 使用 Plotly Express 创建柱状图
fig = px.bar(df, x=df.index, y='time', title="Time Distribution Across Different Algorithms",
             labels={'time': 'Execution Time'}, color='group', barmode='group')
fig.update_layout(
    bargap=0.2,  # 控制柱子之间的间隙，数值范围 [0, 1]，值越小柱子之间的间隙越小
    bargroupgap=0  # 控制分组柱子之间的间隙，数值范围 [0, 1]，值越小柱子之间的间隙越小
)
# 显示图表
fig.show()
