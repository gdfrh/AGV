from Config import *
from order import *
import random


class Arm:
    def __init__(self, work_name, unit_numbers, total_machines, machine_power,num_orders):
        # 初始化生产区名称和单元数
        self.work_name = work_name
        self.unit_numbers = unit_numbers
        self.total_machines = total_machines  # 总机器数量
        self.machine_power = machine_power  # 每台机器的功率
        self.machines_count = {}  # 创建字典来存储每个生产单元的机器数
        self.order_manager = OrderManager(work_name, num_orders)
        self.orders = self.order_manager.get_orders()  # 调用 OrderManager来显示订单
        print(self.orders)
        """还没用到"""
        # self.work_status = {}  # 用来判断生产区是否在工作
        self._initialize_cells()  # 初始化生产区的机器数

    def _initialize_cells(self):
        """初始化生产区及其对应的机器数"""
        for zone, unit_count in zip(self.work_name, self.unit_numbers):
            self.machines_count[zone] = [0] * unit_count  # 为每个生产单元初始化机器数为 0
            # self.work_status[zone] = False  # 初始时，生产区是空闲的


    def distribute_machines_randomly(self):
        """随机将机器分配到生产单元，并显示每个生产区分配的机器数量"""
        available_units = []  # 存储所有生产单元的可用位置（生产区 + 单元索引）

        # 1. 首先为每个生产单元分配一个机器
        for zone in self.machines_count:
            for i in range(len(self.machines_count[zone])):
                self.machines_count[zone][i] = 1  # 每个单元至少分配1台机器
                available_units.append((zone, i))  # 存储可用单元

        # 剩余机器数量
        remaining_machines = self.total_machines - sum([sum(self.machines_count[zone]) for zone in self.machines_count])

        # 2. 随机分配剩余的机器
        while remaining_machines > 0:
            zone, unit_index = random.choice(available_units)  # 随机选择一个单元
            self.machines_count[zone][unit_index] += 1  # 为该单元分配一台机器
            remaining_machines -= 1  # 剩余机器数减少1

    def display_machine_count(self):
        """返回每个生产区中各单元的机器数，并存储到列表中"""
        machine_count_list = []   # 列表
        for zone, units in self.machines_count.items():
            # 将每个生产区的机器数存储到字典中
            machine_count_list += units  # 每个单元的机器数
            # print(
            #     f"{zone}: {', '.join([str(unit) for unit in units])} 台机器")
        merged_str = ''.join(map(str, machine_count_list))
        # 将连接后的字符串转换为整数
        merged_number = int(merged_str)
        machine_count_list_renew = merged_number
        return machine_count_list_renew

    def calculate_reduction(self, initial_time, initial_power, zone_name, machine_count):
        """
        参数:
        initial_time (float): 初始运行时间
        initial_power (float):初始运行功率
        zone (str): 生产区名称
        machine_count (int): 当前该生产单元的机器数量

        返回:
        float: 调整后的运行时间
        """

        reduction_factor_time = 0  #初始化
        reduction_factor_power = 0
        temp_idx = 0
        for zone in self.work_name:

            if zone == zone_name:
                reduction_factor_time = reduction_factor_time_list[temp_idx]
                reduction_factor_power = reduction_factor_power_list[temp_idx]
                break
            temp_idx += 1

        return initial_time * ((1 - reduction_factor_time )**machine_count)\
              ,initial_power*((1 - reduction_factor_power )**machine_count)

    def calculate_task_energy(self, run_time, run_power, sleep_time, sleep_power):
        """
        计算单个任务的能量消耗

        参数:
        run_time (float): 执行工艺任务的运行时间 (秒)
        run_power (float): 执行工艺任务的运行功率 (瓦特)
        sleep_time (float): 休眠时间 (秒)
        sleep_power (float): 休眠功率 (瓦特)
        machine_count (int): 该生产单元的机器数量

        返回:
        float: 该任务的总能量消耗 (焦耳, J)
        """
        # 单个任务的能量消耗
        task_energy = (run_time * run_power) + (sleep_time * sleep_power)
        return task_energy

    def calculate_processing_time(self):
        """计算生产区的处理时间"""
        return processing_time

    def order_time_and_power(self, order):
        """计算一个订单的处理时间和功率消耗，包括生产区时间和运输时间"""
        order_total_time = 0  # 一个订单完成所需时间
        order_total_power = 0  # 一个订单完成所需功率
        max_machines = []  # 用于存储生产单元中最多的机器臂
        # 遍历订单的每个生产区，计算总时间
        for i in order:
            units = self.machines_count[i]
            max_machines.append(max(units))  # 找到最大的机器臂数量
            #max_index = units.index(max_machines)  # 找到最大值的索引
            """这里准备加上空闲、忙碌状态判断"""

            """"""

            """这里我们先找到最多机器臂的生产单元和机器臂数量，再根据数量对时间和功率的影响修改时间和功率"""
        for i in range(len(order) - 1):
            start_zone = order[i]
            #end_zone = order[i + 1]
            order_total_time_renew, order_total_power_renew = self.calculate_reduction(processing_time['run_time'], processing_time['run_power'], start_zone, max_machines[i])
            order_total_time += order_total_time_renew + processing_time['sleep_time']
            order_total_power += self.calculate_task_energy(order_total_time_renew, order_total_power_renew, processing_time['sleep_time'], processing_time['sleep_power']) * max_machines[i]
            # 2. 计算运输时间
            # transport_time = calculate_transport_time(start_zone, end_zone)
            # total_time += transport_time  # 加上运输时间

        return order_total_time, order_total_power


    def function_1(self, sequence):  # 由序列改变字典，用于使用交叉变异修改机器臂分配后计算时间
        """初始化"""
        for zone, unit_count in zip(self.work_name, self.unit_numbers):
            self.machines_count[zone] = [0] * unit_count  # 为每个生产单元初始化机器数为 0

        """序列反填充"""
        sequence_idx = 0  # 追踪当前序列中的索引
        # 使用序列填充机器臂数量
        for zone in self.machines_count:
            for i in range(len(self.machines_count[zone])):
                if sequence_idx < len(list(str(sequence))):
                    # 将 sequence 中的数字逐个赋值到生产区的机器臂数量
                    # 把每个元素转换为字符串，然后逐个字符赋值
                    machines = list(str(sequence))  # 将数字转为字符串
                    self.machines_count[zone][i] = int(machines[sequence_idx])  # 给当前生产单元赋值机器数量
                    sequence_idx += 1

        """任务的指定,需要修改"""
        # tasks = {}
        #
        # for zone, unit_count in zip(self.work_name,self.unit_numbers):
        #     tasks[zone] = []
        #     for _ in range(unit_count):
        #         # 为每个单元生成随机任务列表
        #         tasks[zone].append([self.generate_task() for _ in range(3)])  # 每个生产单元生成3个任务

        """能量，时间计算"""
        total_time, total_power = 0, 0
        for order in self.orders:
            """计算每个订单的消耗"""
            total_time_order, total_power_order = self.order_time_and_power(order)
            total_time += total_time_order
            total_power += total_power_order
        return total_time, total_power
        # total_energy, total_time = 0, 0
        # for zone in self.work_name:
        #     for unit_index in range(len(self.machines_count[zone])):
        #         # 获取该生产单元的机器数量
        #         machine_count = self.machines_count[zone][unit_index]
        #
        #         for task in tasks[zone][unit_index]:
        #             run_time = task['run_time']
        #             run_power = task['run_power']
        #             sleep_time = task['sleep_time']
        #             sleep_power = task['sleep_power']
        #             run_time_renew, run_power_renew = self.calculate_reduction(run_time, run_power, zone, machine_count)
        #
        #             # 计算该任务的能量消耗
        #             task_energy = self.calculate_task_energy(run_time_renew, run_power_renew, sleep_time, sleep_power) * machine_count
        #             total_energy += task_energy
        #             total_time += run_time_renew + sleep_time
        #
        # return total_energy, total_time
"""现在所计算出来的都是一个订单的时间和功率并且是用了最多机器臂情况下的结果，并且时间是累加计算的，应该得计算最后结束就行"""

