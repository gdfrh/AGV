import glob
import pickle
import pandas as pd
import numpy as np
import os
from collections import defaultdict


def process_grouped_data(folder_path, output_prefix, target_folder):
    """按文件名前缀分组处理数据，生成多个Excel文件"""
    # 获取所有pkl文件并按前缀分组
    file_paths = glob.glob(os.path.join(folder_path, '*.pkl'))
    file_groups = defaultdict(list)

    for fp in file_paths:
        filename = os.path.basename(fp)
        # 提取前缀（例如 "1" from "1_0.pkl"）
        prefix = filename.split('_')[0]
        file_groups[prefix].append(fp)

    # 遍历每个分组
    for prefix, files in file_groups.items():
        all_files_data = []

        # 处理组内每个文件
        for file_path in files:
            with open(file_path, 'rb') as file:
                loaded_data = pickle.load(file)

                # 数据统计计算
                energy = loaded_data['energy']
                time = loaded_data['time']

                file_stats = {
                    'Filename': os.path.basename(file_path),
                    'Energy_Mean': np.mean(energy),
                    'Energy_Best': np.min(energy),
                    'Energy_Variance': np.var(energy),
                    'Time_Mean': np.mean(time),
                    'Time_Best': np.min(time),
                    'Time_Variance': np.var(time)
                }
                all_files_data.append(file_stats)

        # 生成DataFrame
        df = pd.DataFrame(all_files_data)
        columns_order = [
            'Filename',
            'Energy_Mean', 'Energy_Best', 'Energy_Variance',
            'Time_Mean', 'Time_Best', 'Time_Variance'
        ]
        df = df[columns_order]

        # 输出信息
        print(f"\nProcessing prefix group: {prefix}")
        print(f"Found {len(files)} files in group")
        print(df.head())

        # 保存Excel
        os.makedirs(target_folder, exist_ok=True)
        output_name = f"{output_prefix}_{prefix}.xlsx"
        output_path = os.path.join(target_folder, output_name)
        df.to_excel(output_path, index=False)
        print(f"Saved group {prefix} to: {output_path}")


process_grouped_data('tenth_compare/50_15_15', 'tenth_comparison', 'results')
