import pickle
import pandas as pd
import glob
import numpy as np
import os


# 处理文件夹路径数据的函数
def process_folder_data(folder_path, output_excel_name, target_folder):
    # 获取文件夹中的所有.pkl文件
    file_paths = glob.glob(f'{folder_path}/*.pkl')

    energy_table = []
    time_table = []

    # 遍历所有文件
    for idx, file_path in enumerate(file_paths):
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)
            # 获取文件中的数据
            energy = loaded_data['energy']
            for i in range(len(energy)):
                energy_table.append(energy[i])
            time = loaded_data['time']
            for i in range(len(time)):
                time_table.append(time[i])

    # 计算 energy 和 time 的统计量
    energy_mean = np.mean(energy_table)
    energy_min = np.min(energy_table)
    energy_var = np.var(energy_table)

    time_mean = np.mean(time_table)
    time_min = np.min(time_table)
    time_var = np.var(time_table)

    # 将统计量保存为字典
    data = {
        'Metric': ['Mean', 'Best', 'Variance'],
        'Energy': [energy_mean, energy_min, energy_var],
        'Time': [time_mean, time_min, time_var]
    }

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 输出到控制台
    print(f"Data from folder: {folder_path}")
    print(df)

    # 保存为 Excel 文件到指定文件夹
    output_file_path = os.path.join(target_folder, output_excel_name)
    df.to_excel(output_file_path, index=False)

process_folder_data('greedy_repair_t_e', 'energy_time_table1.xlsx', 'greedy_repair_t_e')
process_folder_data('regret_repair_t_e', 'energy_time_table2.xlsx', 'regret_repair_t_e')

