# import random
# from collections import deque
# import random
# import time
#
# import random
#
# class Order:
#     def __init__(self, order_id, all_zones):
#         self.order_id = order_id  # 订单ID
#         self.zones = self.generate_order_zones(all_zones)  # 订单所需的生产区顺序
#         self.current_zone_index = 0  # 当前订单正在处理的生产区索引
#
#     def generate_order_zones(self, all_zones):
#         """
#         生成一个订单所需的生产区顺序。
#         - 订单总是从生产区 0 开始
#         - 后续的生产区可以从剩余的区中随机选择
#         """
#         zones = [0]  # 订单总是从生产区 0 开始
#         remaining_zones = list(range(1, len(all_zones)))  # 剩余的生产区
#         random.shuffle(remaining_zones)  # 随机选择后续的生产区顺序
#         zones.extend(remaining_zones)
#         return zones
#
#     def next_zone(self):
#         """返回下一个生产区"""
#         if self.current_zone_index < len(self.zones):
#             zone = self.zones[self.current_zone_index]
#             self.current_zone_index += 1
#             return zone
#         return None  # 所有生产区都已处理完
#
#     def get_order_details(self):
#         return {
#             "order_id": self.order_id,
#             "zones": self.zones
#         }
# class OrderManager:
#     def __init__(self, all_zones, num_orders):
#         self.all_zones = all_zones  # 所有的生产区
#         self.orders = self.generate_orders(num_orders)  # 生成订单
#
#     def generate_orders(self, num_orders):
#         """生成多个订单"""
#         orders = []
#         for i in range(num_orders):
#             order = Order(i, self.all_zones)
#             orders.append(order)
#         return orders
#
#     def print_orders(self):
#         for order in self.orders:
#             details = order.get_order_details()
#             print(f"Order ID: {details['order_id']}, Production Zones: {details['zones']}")
# class Arm:
#     def __init__(self, work_name, unit_numbers, total_machines, machine_power):
#         self.work_name = work_name
#         self.unit_numbers = unit_numbers
#         self.total_machines = total_machines
#         self.machine_power = machine_power
#         self.machines_count = {}  # 存储每个生产区的机器数量
#         self.work_status = {}  # 存储生产区是否在工作
#         self._initialize_cells()
#
#     def _initialize_cells(self):
#         """初始化生产区及其机器数量"""
#         for zone in self.work_name:
#             self.machines_count[zone] = 0  # 初始化每个生产区的机器数量为 0
#             self.work_status[zone] = False  # 初始时，生产区是空闲的
#
#     def assign_task(self, zone):
#         """为生产区分配任务并设置为工作状态"""
#         if self.work_status[zone]:  # 如果生产区已经在工作，返回False表示无法分配
#             return False
#         self.work_status[zone] = True  # 设置该生产区为工作状态
#         print(f"Assigning task to {zone}")
#         return True
#
#     def complete_task(self, zone):
#         """任务完成后，设置生产区为空闲状态"""
#         self.work_status[zone] = False
#         print(f"Task completed in {zone}, production area is now free.")
#
# class Scheduler:
#     def __init__(self, work_name, unit_numbers, total_machines, machine_power, num_orders):
#         self.arm = Arm(work_name, unit_numbers, total_machines, machine_power)
#         self.order_manager = OrderManager(work_name, num_orders)
#         self.order_queue = deque(self.order_manager.orders)  # 使用队列存储订单
#
#     def process_orders(self):
#         """处理所有订单"""
#         while self.order_queue:
#             order = self.order_queue.popleft()  # 获取当前订单
#             print(f"Processing Order {order.order_id}")
#             while True:
#                 # 获取当前订单需要的生产区
#                 zone = order.next_zone()
#                 if zone is None:
#                     print(f"Order {order.order_id} completed!")
#                     break  # 订单完成
#
#                 # 尝试分配任务
#                 if not self.arm.assign_task(zone):
#                     print(f"Production zone {zone} is busy. Order {order.order_id} is waiting.")
#                     self.order_queue.append(order)  # 如果生产区繁忙，重新将订单放回队列中等待
#                     break  # 当前订单停止处理，等待下一轮
#
#                 # 模拟任务处理（例如：等待生产区任务完成）
#                 self.arm.complete_task(zone)  # 模拟任务完成，释放生产区
#
# # 示例：创建生产区、订单和调度系统
# work_name = ["组装区", "铸造区", "清洗区", "包装区","焊接区", "喷漆区", "配置区"]
# unit_numbers = [2, 4, 3, 2, 3, 2, 4]  # 每个上半生产区的单元数
# total_machines = 50
# machine_power = 100
# num_orders = 5
#
# scheduler = Scheduler(work_name, unit_numbers, total_machines, machine_power, num_orders)
# scheduler.process_orders()
# a=[0, 6, 2, 1, 5, 4, 3]
# print(len(a))
# import time
# a=time.time()
#
# time.sleep(2)
# b=time.time()
# print(a,b)
import bisect
sorted_list = [1,2,3,5,5,5,7,8,9,10]
target_agv_count = 5
pos = bisect.bisect_left(sorted_list, target_agv_count)
if pos > 0:
    smaller_value = sorted_list[pos - 1]
else:
    smaller_value = None  # 如果 target 是最小值，则没有比它小的值

pos = bisect.bisect_right(sorted_list, target_agv_count)
if pos < len(sorted_list):
    larger_value = sorted_list[pos]
else:
    larger_value = None  # 如果 target 是最大值，则没有比它大的值
print(smaller_value, larger_value)