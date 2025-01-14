import time
import copy
import random
from Config import *
from NSGA2 import main_loop
from robot_arm import Arm
from car import Car

"""我将整体的整合代码"""


class Schedule:
    def __init__(self, work_name, total_machines, machine_power, num_orders, zone_requirements):
        """初始化调度类"""
        self.unit_numbers = self.get_max_units(zone_requirements, total_machines)  # 获取最多生产单元数量
        self.arm = Arm(work_name, self.unit_numbers, total_machines, machine_power, num_orders)  # 创建 Arm 类实例
        self.car = Car(work_name, self.unit_numbers, total_machines)  # 创建 Car 类实例
        self.orders = self.arm.orders  # 获取从 OrderManager 获取到的订单

    def get_max_units(self, zone_requirements, total_machines):
        # 初始化每个生产区的生产单元数和剩余机器臂
        production_units = {}
        remaining_machines = total_machines

        # 第一轮给每个生产区分配一个生产单元
        for zone, min_machines in zone_requirements:
            # 消耗掉每个生产区的初始机器臂
            production_units[zone] = 1
            remaining_machines -= min_machines

        # 然后根据剩余的机器臂来计算每个生产区最多能有多少个生产单元
        for zone, min_machines in zone_requirements:
            if remaining_machines >= min_machines:
                # 对于剩余的机器臂，计算最多能分配的生产单元数
                max_additional_units = remaining_machines // min_machines
                production_units[zone] += max_additional_units

        # 将每个生产区的最多生产单元的数量按照顺序存储到列表中
        # 如果你希望按生产区名称的顺序排列
        max_units_list = [production_units[zone] for zone, _ in zone_requirements]

        return max_units_list

    def arm_random(self):
        # 随机分配所有机器臂，形成一组机器臂的初始解

        while len(machine_counts) < pop_size:
            self.arm.distribute_machines_randomly()
            """添加初始分布状态"""
            self.arm.unit_states.append(self.unit_numbers)
            new_list = copy.deepcopy(self.arm.display_machine_count())

            """还没用到这个"""
            # energy_count, time_count = init_arm.object_function(new_list)
            #
            # # 保留两位小数
            # energy_count = round(energy_count, 2)
            # time_count = round(time_count, 2)
            #
            # energy_counts.append(energy_count)
            # time_counts.append(time_count)
            """还没用到上面"""
            if new_list not in machine_counts:
                machine_counts.append(new_list)  # 列表形式记录机器臂数量
    def arm_loop(self):
        """实现机器臂的 NSGA-II算法来优化机器臂数量"""
        v1, v2 = main_loop(pop_size, max_gen, machine_counts, self.arm)