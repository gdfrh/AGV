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
zone_units =[0,1,2,3,4,5]
zone_units.append(6)
zone_units = sorted(zone_units, reverse=True)
print(zone_units)
expanded_list = [0] + zone_units + [1]
expanded_list = sorted(expanded_list, reverse=True)
print(expanded_list)
