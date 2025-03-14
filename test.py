# # # # # import random
# # # # # from collections import deque
# # # # # import random
# # # # # import time
# # # # #
# # # # # import random
# # # # #
# # # # # class Order:
# # # # #     def __init__(self, order_id, all_zones):
# # # # #         self.order_id = order_id  # 订单ID
# # # # #         self.zones = self.generate_order_zones(all_zones)  # 订单所需的生产区顺序
# # # # #         self.current_zone_index = 0  # 当前订单正在处理的生产区索引
# # # # #
# # # # #     def generate_order_zones(self, all_zones):
# # # # #         """
# # # # #         生成一个订单所需的生产区顺序。
# # # # #         - 订单总是从生产区 0 开始
# # # # #         - 后续的生产区可以从剩余的区中随机选择
# # # # #         """
# # # # #         zones = [0]  # 订单总是从生产区 0 开始
# # # # #         remaining_zones = list(range(1, len(all_zones)))  # 剩余的生产区
# # # # #         random.shuffle(remaining_zones)  # 随机选择后续的生产区顺序
# # # # #         zones.extend(remaining_zones)
# # # # #         return zones
# # # # #
# # # # #     def next_zone(self):
# # # # #         """返回下一个生产区"""
# # # # #         if self.current_zone_index < len(self.zones):
# # # # #             zone = self.zones[self.current_zone_index]
# # # # #             self.current_zone_index += 1
# # # # #             return zone
# # # # #         return None  # 所有生产区都已处理完
# # # # #
# # # # #     def get_order_details(self):
# # # # #         return {
# # # # #             "order_id": self.order_id,
# # # # #             "zones": self.zones
# # # # #         }
# # # # # class OrderManager:
# # # # #     def __init__(self, all_zones, num_orders):
# # # # #         self.all_zones = all_zones  # 所有的生产区
# # # # #         self.orders = self.generate_orders(num_orders)  # 生成订单
# # # # #
# # # # #     def generate_orders(self, num_orders):
# # # # #         """生成多个订单"""
# # # # #         orders = []
# # # # #         for i in range(num_orders):
# # # # #             order = Order(i, self.all_zones)
# # # # #             orders.append(order)
# # # # #         return orders
# # # # #
# # # # #     def print_orders(self):
# # # # #         for order in self.orders:
# # # # #             details = order.get_order_details()
# # # # #             print(f"Order ID: {details['order_id']}, Production Zones: {details['zones']}")
# # # # # class Arm:
# # # # #     def __init__(self, work_name, unit_numbers, total_machines, machine_power):
# # # # #         self.work_name = work_name
# # # # #         self.unit_numbers = unit_numbers
# # # # #         self.total_machines = total_machines
# # # # #         self.machine_power = machine_power
# # # # #         self.machines_count = {}  # 存储每个生产区的机器数量
# # # # #         self.work_status = {}  # 存储生产区是否在工作
# # # # #         self._initialize_cells()
# # # # #
# # # # #     def _initialize_cells(self):
# # # # #         """初始化生产区及其机器数量"""
# # # # #         for zone in self.work_name:
# # # # #             self.machines_count[zone] = 0  # 初始化每个生产区的机器数量为 0
# # # # #             self.work_status[zone] = False  # 初始时，生产区是空闲的
# # # # #
# # # # #     def assign_task(self, zone):
# # # # #         """为生产区分配任务并设置为工作状态"""
# # # # #         if self.work_status[zone]:  # 如果生产区已经在工作，返回False表示无法分配
# # # # #             return False
# # # # #         self.work_status[zone] = True  # 设置该生产区为工作状态
# # # # #         print(f"Assigning task to {zone}")
# # # # #         return True
# # # # #
# # # # #     def complete_task(self, zone):
# # # # #         """任务完成后，设置生产区为空闲状态"""
# # # # #         self.work_status[zone] = False
# # # # #         print(f"Task completed in {zone}, production area is now free.")
# # # # #
# # # # # class Scheduler:
# # # # #     def __init__(self, work_name, unit_numbers, total_machines, machine_power, num_orders):
# # # # #         self.arm = Arm(work_name, unit_numbers, total_machines, machine_power)
# # # # #         self.order_manager = OrderManager(work_name, num_orders)
# # # # #         self.order_queue = deque(self.order_manager.orders)  # 使用队列存储订单
# # # # #
# # # # #     def process_orders(self):
# # # # #         """处理所有订单"""
# # # # #         while self.order_queue:
# # # # #             order = self.order_queue.popleft()  # 获取当前订单
# # # # #             print(f"Processing Order {order.order_id}")
# # # # #             while True:
# # # # #                 # 获取当前订单需要的生产区
# # # # #                 zone = order.next_zone()
# # # # #                 if zone is None:
# # # # #                     print(f"Order {order.order_id} completed!")
# # # # #                     break  # 订单完成
# # # # #
# # # # #                 # 尝试分配任务
# # # # #                 if not self.arm.assign_task(zone):
# # # # #                     print(f"Production zone {zone} is busy. Order {order.order_id} is waiting.")
# # # # #                     self.order_queue.append(order)  # 如果生产区繁忙，重新将订单放回队列中等待
# # # # #                     break  # 当前订单停止处理，等待下一轮
# # # # #
# # # # #                 # 模拟任务处理（例如：等待生产区任务完成）
# # # # #                 self.arm.complete_task(zone)  # 模拟任务完成，释放生产区
# # # # #
# # # # # # 示例：创建生产区、订单和调度系统
# # # # # work_name = ["组装区", "铸造区", "清洗区", "包装区","焊接区", "喷漆区", "配置区"]
# # # # # unit_numbers = [2, 4, 3, 2, 3, 2, 4]  # 每个上半生产区的单元数
# # # # # total_machines = 50
# # # # # machine_power = 100
# # # # # num_orders = 5
# # # # #
# # # # # scheduler = Scheduler(work_name, unit_numbers, total_machines, machine_power, num_orders)
# # # # # scheduler.process_orders()
# # # # # a=[0, 6, 2, 1, 5, 4, 3]
# # # # # print(len(a))
# # # # # import time
# # # # # a=time.time()
# # # # #
# # # # # time.sleep(2)
# # # # # b=time.time()
# # # # # print(a,b)
# # # # # import bisect
# # # # # sorted_list = [1,2,3,5,5,5,7,8,9,10]
# # # # # target_agv_count = 5
# # # # # pos = bisect.bisect_left(sorted_list, target_agv_count)
# # # # # if pos > 0:
# # # # #     smaller_value = sorted_list[pos - 1]
# # # # # else:
# # # # #     smaller_value = None  # 如果 target 是最小值，则没有比它小的值
# # # # #
# # # # # pos = bisect.bisect_right(sorted_list, target_agv_count)
# # # # # if pos < len(sorted_list):
# # # # #     larger_value = sorted_list[pos]
# # # # # else:
# # # # #     larger_value = None  # 如果 target 是最大值，则没有比它大的值
# # # # # print(smaller_value, larger_value)
# # # # import random
# # # # import math
# # # #
# # # #
# # # # # 生成随机城市坐标
# # # # def generate_cities(n):
# # # #     return [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n)]
# # # #
# # # #
# # # # # 计算城市之间的欧几里得距离
# # # # def calculate_distance(city1, city2):
# # # #     return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
# # # #
# # # #
# # # # # 计算总路径长度
# # # # def total_distance(route, cities):
# # # #     return sum(calculate_distance(cities[route[i]], cities[route[i + 1]]) for i in range(len(route) - 1)) + \
# # # #         calculate_distance(cities[route[-1]], cities[route[0]])
# # # #
# # # #
# # # # # 生成一个随机的初始解
# # # # def generate_initial_solution(n):
# # # #     return random.sample(range(n), n)
# # # #
# # # #
# # # # # 扰动操作：交换路径中的两个城市
# # # # def swap(route):
# # # #     i, j = random.sample(range(len(route)), 2)
# # # #     route[i], route[j] = route[j], route[i]
# # # #
# # # #
# # # # # 扰动操作：反转路径的一部分
# # # # def reverse(route):
# # # #     i, j = sorted(random.sample(range(len(route)), 2))
# # # #     route[i:j + 1] = reversed(route[i:j + 1])
# # # #
# # # #
# # # # # 选择邻域操作
# # # # def select_neighborhood():
# # # #     return random.choice([swap, reverse])
# # # #
# # # #
# # # # # ALNS算法
# # # # def alns(cities, max_iterations=1000, temperature=1.0, cooling_rate=0.995):
# # # #     n = len(cities)
# # # #     current_solution = generate_initial_solution(n)
# # # #     best_solution = current_solution
# # # #     best_distance = total_distance(best_solution, cities)
# # # #
# # # #     for iteration in range(max_iterations):
# # # #         # 随机选择一个邻域操作
# # # #         neighborhood = select_neighborhood()
# # # #
# # # #         # 生成新解
# # # #         new_solution = current_solution[:]
# # # #         neighborhood(new_solution)
# # # #
# # # #         # 计算新解的总距离
# # # #         new_distance = total_distance(new_solution, cities)
# # # #
# # # #         # 接受准则：模拟退火
# # # #         if new_distance < best_distance:
# # # #             current_solution = new_solution
# # # #             best_solution = new_solution
# # # #             best_distance = new_distance
# # # #         else:
# # # #             acceptance_prob = math.exp(-(new_distance - total_distance(current_solution, cities)) / temperature)
# # # #             if random.random() < acceptance_prob:
# # # #                 current_solution = new_solution
# # # #
# # # #         # 降低温度
# # # #         temperature *= cooling_rate
# # # #
# # # #     return best_solution, best_distance
# # # #
# # # #
# # # # # 测试代码
# # # # if __name__ == "__main__":
# # # #     n = 20  # 城市数量
# # # #     cities = generate_cities(n)
# # # #
# # # #     best_solution, best_distance = alns(cities)
# # # #     print("Best Solution:", best_solution)
# # # #     print("Best Distance:", best_distance)
# # #
# # # # # 假设 total_time_list 存储的是总时间
# # # # total_time_list = [10, 25, 15, 30, 5, 60, 35, 20, 45, 55]
# # # #
# # # # # 假设 top_20_percent_indices 存储的是需要删除的索引列表
# # # # top_20_percent_indices = [5, 9, 6]  # 要删除的索引
# # # #
# # # # # 按照从大的索引到小的索引删除元素
# # # # for index in sorted(top_20_percent_indices, reverse=True):
# # # #     del total_time_list[index]
# # # #
# # # # print(total_time_list)  # 输出：[10, 25, 15, 30, 5, 35, 20, 45]
# # # import random
# # #
# # # # # 假设 total_time_list 存储的是总时间
# # # # total_time_list = [10, 25, 15, 30, 5, 60, 35, 20, 45, 55]
# # # #
# # # # # 假设 top_20_percent_indices 存储的是需要删除的索引列表
# # # # top_20_percent_indices = [5, 9, 6]  # 要删除的索引
# # # #
# # # # # 存储删除的元素
# # # # removed_elements = []
# # # #
# # # # # 按照从大的索引到小的索引删除元素
# # # # for index in sorted(top_20_percent_indices, reverse=True):
# # # #     removed_elements.append(total_time_list[index])  # 存储删除的元素
# # # #     del total_time_list[index]  # 删除元素
# # # #
# # # # # 打印删除后的 total_time_list
# # # # print("After deletion:", total_time_list)
# # # #
# # # # # 随机顺序将删除的元素插入到列表尾部
# # # # random.shuffle(removed_elements)  # 随机打乱删除的元素
# # # #
# # # # # 将删除的元素插入到列表尾部
# # # # total_time_list.extend(removed_elements)
# # # #
# # # # # 打印最终结果
# # # # print("After insertion:", total_time_list)
# # # # orders = [[1,2],[1,3],[1,4],[1,5]]
# # # # print(len(orders))
# # # # import numpy as np
# # # # similarity_matrix = np.zeros((2, 2))
# # # # similarity_matrix[0, 0] = 1
# # # # similarity_matrix[1, 1] = 0
# # # # similarity_matrix[1, 0] = 2
# # # # similarity_matrix[0, 1] = 3
# # # # n = similarity_matrix.sum(axis=1)
# # # # print(n)
# # # # new_order = [1,2,3,4,5]
# # # # indices = [0,2,1]
# # # # for index in sorted(indices, reverse=True):  # reverse=True 确保从后往前删除
# # # #     del new_order[index]
# # # # print(new_order)
# # # # zones = [0, 1]
# # # #
# # # # # 剩余的生产区（不包含0号区）
# # # # # remaining_zones = list(range(2, 6))
# # # # # max_zones = random.randint(1, 7)
# # # # #
# # # # # num_zones_to_select = min(max_zones - 1, len(remaining_zones))  # 除了0、1号区外，最多选择的数量
# # # # #
# # # # # # 随机选择剩余的生产区
# # # # # random.shuffle(remaining_zones)
# # # # # selected_zones = remaining_zones[:num_zones_to_select]
# # # # #
# # # # # # 将选择的生产区加入订单的生产区顺序
# # # # # zones.extend(selected_zones)
# # # # a = {}
# # # # a[1] = [1]
# # # # a[1].append(2)
# # # # print(a)
# # # # a = [1,2,3,4,5]
# # # # xxxx = a[2:3]
# # # # print(xxxx)
# # # # import itertools
# # # # regret_matching_operator = [[0, 1, 2, 3, 4, 6],[0, 1, 2, 4, 5, 6]]
# # # # permutations_dict = {}
# # # # for idx, lst in enumerate(regret_matching_operator):
# # # #     # 从索引2到倒数第二个元素（不包括最后的6）
# # # #     disruption = lst[2:-1]
# # # #     # 生成xxxx的所有排列
# # # #     permutations = list(itertools.permutations(disruption))
# # # #     permutations = [[0, 1] + list(perm) + [6] for perm in permutations]
# # # #     print(permutations)
# # # #     # 存储排列,现在所有的排列情况都存在了字典中
# # # #     permutations_dict[idx] = permutations
# # # import itertools
# # #
# # # # 定义示例字典
# # # regret_matching_operator = [
# # #     [0, 1, 2, 3, 4, 5, 6],
# # # ]
# # #
# # # permutations_dict = {}
# # #
# # # # 生成排列并存储在字典中
# # # for idx, lst in enumerate(regret_matching_operator):
# # #     disruption = lst[2:-1]
# # #     permutations = list(itertools.permutations(disruption))
# # #     permutations = [[0, 1] + list(perm) + [6] for perm in permutations]
# # #     permutations_dict[idx] = permutations
# # #
# # # # 遍历字典，并对每个排列列表进行操作
# # # for key, permutations in permutations_dict.items():
# # #     print(f"Processing list {key} permutations:")
# # #     for perm in permutations:
# # #         # 这里可以进行你需要的任何操作
# # #         print(perm)  # 打印排列
# # #
# # # import numpy as np
# # # array = [1,3,5]
# # # brrby = [1,5,9]
# # # row_vectors = []
# # # row_vectors.append(array)
# # # row_vectors.append(brrby)
# # # print(row_vectors)
# # # # 将所有行向量拼接成矩阵
# # # matrix = np.vstack(row_vectors)
# # # print(matrix)
# # # # i = 2
# # # # new_order = [1,2,3]
# # # # order_sequence = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
# # # # new_sequence = order_sequence[:i] + [new_order] + order_sequence[i:]
# # # # print(new_sequence)
# # # # scores_array = np.zeros(len(order_sequence) + 1)
# # # # print(scores_array)
# # # work_name_up = ['组装区', '铸造区', '清洗区', '包装区']
# # # work_name_down = ['焊接区', '喷漆区', '配置区']
# # # work_name = work_name_up + work_name_down
# # # print(work_name[-1])
# # def find_key_and_list_for_index(data_dict, index):
# #     current_index = 0
# #     for key, lists in data_dict.items():
# #         for lst in lists:
# #             if current_index == index:
# #                 return key, lst  # 返回找到的键和列表
# #             current_index += 1
# #     return None, None  # 如果没有找到，返回None
# #
# # # 示例使用
# # data_dict = {
# #     'key1': [['list1'], ['list2'], ['list3']],
# #     'key2': [['list4'], ['list5'], ['list6'], ['list7']],
# #     'key3': [['list8'], ['list9'], ['list10'], ['list11'], ['list12']]
# # }
# #
# # index_to_find = 11
# # found_key, found_list = find_key_and_list_for_index(data_dict, index_to_find)
# # print(f"索引 {index_to_find} 对应的键是：{found_key}")
# # print(f"该键对应的列表是：{found_list}")
# data_list = [
#     {'组装区': 10, '铸造区': 20, '清洗区': 30},
#     {'组装区': 40, '铸造区': 50, '清洗区': 60},
#     {'组装区': 70, '铸造区': 80, '清洗区': 90}
# ]
# import pandas as pd
# import plotly.express as px
# # 从字典列表创建 DataFrame
# df = pd.DataFrame(data_list)
#
# print(df)
# fig = px.scatter(df, x='x', y='y', title="Energy vs. Time Scatter Plot",
#                  hover_data=['组装区', '铸造区', '清洗区'])
# # 显示图表
# fig.show()
#
import re

# 假设这是您的数字
num1 = 440000000222200000000000002220000000000000030000000000320000000000000002200000000000000043333000000
num2 = 432000000322200000000000005220000000000000060000000000620000000000000002200000000000000043333000000
# 将数字转换为字符串
num_str1 = str(num1)
num_str2 = str(num2)

# 使用正则表达式查找连续的非零数字和后面跟随的所有零
parts1 = re.findall(r'[1-9]+0*', num_str1)
parts2 = re.findall(r'[1-9]+0*', num_str2)

# 假设 idx 是已定义的索引列表，需要交换这些索引位置的数字
idx_list = [1, 3]  # 例如，我们要交换第2和第4个字符位置的数字

new_parts1 = []
new_parts2 = []

for i in range(len(parts1)):  # 遍历每一个部分
    # 将字符串中的每个字符转换为整数，存入新列表
    expanded1_list = [int(digit) for digit in parts1[i]]
    expanded2_list = [int(digit) for digit in parts2[i]]

    # 对于每个要交换的索引，执行交换操作
    for idx in idx_list:
        if idx < len(expanded1_list) and idx < len(expanded2_list):  # 确保索引不会超出列表长度
            expanded1_list[idx], expanded2_list[idx] = expanded2_list[idx], expanded1_list[idx]

    # 将修改后的 expanded_list 中的数字重新组合成一个新的字符串
    new_str1 = ''.join(map(str, expanded1_list))
    new_str2 = ''.join(map(str, expanded2_list))

    # 添加到新列表
    new_parts1.append(new_str1)
    new_parts2.append(new_str2)

print("New parts1:", new_parts1)
print("New parts2:", new_parts2)

new_parts2 = ['442000000', '32220000000000000', '52200000000000000', '60000000000', '62000000000000000', '22000000000000000', '43333000000']

# 使用 join() 方法将列表中的所有字符串连接成一个单一的字符串
combined_number = ''.join(new_parts2)

print("Combined Number:", combined_number)
