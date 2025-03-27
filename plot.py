import pickle
from Config import *
import plotly.express as px
import pandas as pd
import glob

# 获取所有.pickle文件的路径
file_paths = glob.glob('*.pkl')  # 假设所有文件都在当前目录

# 用于存储所有的数据
combined_data = {
    'energy': [],
    'time': [],
    'agv_distribution': [],
    'order': [],
    'distributions_dict': [],
    'group': []  # 用于存储每个数据来源的 group 信息
}

arm_distributions = []

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

    # 处理 distributions_dicts，可能是字典格式的，转化为可展平的格式
    for dic in distributions_dicts:
        arm_distributions.append(dic)

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
