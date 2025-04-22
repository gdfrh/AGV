import glob
import pickle
import pandas as pd
import numpy as np
import os


def process_folder_data(folder_path, output_excel_name, target_folder):
    # 获取文件夹中的所有.pkl文件
    file_paths = glob.glob(os.path.join(folder_path, '*.pkl'))

    all_files_data = []  # 存储所有文件统计结果

    # 遍历所有文件
    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)

            # 提取数据
            energy = loaded_data['energy']
            time = loaded_data['time']

            # 计算统计量
            file_stats = {
                'Filename': os.path.basename(file_path),  # 记录文件名
                'Energy_Mean': np.mean(energy),
                'Energy_Best': np.min(energy),
                'Energy_Variance': np.var(energy),
                'Time_Mean': np.mean(time),
                'Time_Best': np.min(time),
                'Time_Variance': np.var(time)
            }
            all_files_data.append(file_stats)

    # 转换为DataFrame
    df = pd.DataFrame(all_files_data)

    # 重新排列列顺序
    columns_order = [
        'Filename',
        'Energy_Mean', 'Energy_Best', 'Energy_Variance',
        'Time_Mean', 'Time_Best', 'Time_Variance'
    ]
    df = df[columns_order]

    # 输出到控制台
    print(f"\nProcessing folder: {folder_path}")
    print(f"Found {len(file_paths)} files")
    print(df.head())

    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)

    # 保存为Excel文件
    output_path = os.path.join(target_folder, output_excel_name)
    df.to_excel(output_path, index=False)
    print(f"Saved to: {output_path}")


# 使用示例
process_folder_data('greedy_repair_t_e', 'greedy_stats.xlsx', 'results')
process_folder_data('regret_repair_t_e', 'regret_stats.xlsx', 'results')