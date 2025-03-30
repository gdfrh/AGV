# import pickle
#
# # 假设这些是您要保存的数据
# energy_pic = [1, 2, 3, 4]
# time_pic = [4, 3, 2, 1]
# agv_distributions = [[1, 0], [2, 1], [1, 1], [0, 0]]
# order_distributions = [[1, 2], [2, 3], [3, 4], [4, 5]]
# distributions_dicts = [{'zone1': [1, 2]}, {'zone2': [2, 3]}]
#
# # 将数据存储到文件
# data = {
#     'energy': energy_pic,
#     'time': time_pic,
#     'agv_distribution': agv_distributions,
#     'order': order_distributions,
#     'distributions_dict': distributions_dicts
# }
#
# with open('data.pkl', 'wb') as file:
#     pickle.dump(data, file)
# import random
#
# # 示例数据：总数和各生产单元机器臂的初始分布
# total_machines = 25  # 假设总机器臂数量为25
# parts = [5, 5, 5, 5, 5]  # 假设有5个生产单元，每个生产单元初始分配的机器臂数量
#
# # 确保总数量不超过总机器臂数量
# current_sum = sum(expanded_list)
# remaining_machines = total_machines - current_sum
#
# # 将剩余的机器臂数量分配到已分配的机器臂数量中
# for i in range(remaining_machines):
#     # 随机选择一个生产单元，增加机器臂
#     random_index = random.choice(range(len(parts)))
#     expanded_list[random_index] += 1
# import random
#
# list = [0,0,0]
# print(len(list))
# print(random.random(),random.random(),random.random())
# import random
# zone_units =[0,1,2,3,4,5]
# unit_to_remove = random.choice(zone_units)
# print(unit_to_remove)
# zone_units.remove(unit_to_remove)
# print(zone_units)
# zone_units =[0,1,2,3,4,5]
# zone_units.append(6)
# zone_units = sorted(zone_units, reverse=True)
# print(zone_units)
# expanded_list = [0] + zone_units + [1]
# expanded_list = sorted(expanded_list, reverse=True)
# print(expanded_list)
# arm_distributions={'a':1,'b':2,'c':3,'d':4,'e':5}
# for zone in arm_distributions:
#     print(zone)
# 定义一个新字典
import random
from Config import *
arm_distributions = {}
for zone, unit_count in zip(work_name, [3,3,3,3,3,3,3]):
    arm_distributions[zone] = [0] * unit_count  # 为每个生产单元初始化机器数为 0
    # 1. 根据需求给每个生产区分配机器
remaining_machines = total_machines  # 总机器数量
for zone, min_machines in zone_requirements:  # zone_requirements 中保存每个生产区需要的最小机器数量
    # 获取该生产区的单元数
    units = len(arm_distributions[zone])
    obj_units = random.randint(0, units - 1)
    # 先给每个生产单元分配最低机器臂数量
    arm_distributions[zone][obj_units] = min_machines
    remaining_machines -= min_machines
# 2. 随机分配机器
while remaining_machines > 0:
    zone = random.choice(work_name)  # 随机选择一个生产区
    unit_index = random.randint(0, len(arm_distributions[zone]) - 1)  # 随机选择一个生产单元
    # 分配每个生产区需要的机器数，而不是逐个机器分配
    """如果机器臂数量为0，则分配最低要求数"""
    if arm_distributions[zone][unit_index] == 0:
        min_required_machines = zone_requirements[work_name.index(zone)][1]  # 获取该生产区的最小机器需求
        if remaining_machines >= min_required_machines:
            arm_distributions[zone][unit_index] = min_required_machines
            remaining_machines -= min_required_machines
        """如果已分配机器臂，就多分配1个"""
    elif arm_distributions[zone][unit_index] != 0:
        arm_distributions[zone][unit_index] += 1
        remaining_machines -= 1
# 3. 对每个生产区的生产单元按机器数量从大到小排序
for zone in arm_distributions:
    # 排序每个生产区的单元，按机器数量从大到小
    arm_distributions[zone] = sorted(arm_distributions[zone], reverse=True)
machine_count_list = []  # 列表
for zone, units in arm_distributions.items():
    # 将每个生产区的机器数存储到字典中
    machine_count_list += units  # 每个单元的机器数
merged_str = ''.join(map(str, machine_count_list))
# 将连接后的字符串转换为整数
merged_number = int(merged_str)

print(merged_number)