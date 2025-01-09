import time
import copy
import random
from Config import *
from NSGA2 import main_loop
from robot_arm import Arm
from car import Car

"""我将整体的整合代码"""


class Schedule:
    def __init__(self, work_name, unit_numbers, total_machines, machine_power, num_orders):
        """初始化调度类"""
        self.arm = Arm(work_name, unit_numbers, total_machines, machine_power, num_orders)  # 创建 Arm 类实例
        self.car = Car(work_name, unit_numbers, total_machines)  # 创建 Car 类实例
        self.orders = self.arm.orders  # 获取从 OrderManager 获取到的订单

    #initialize
    def arm_random(self):
        # 随机分配所有机器臂，形成一组机器臂的初始解
        for _ in range(pop_size):
            self.arm.distribute_machines_randomly()
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

            machine_counts.append(new_list)  # 列表形式记录机器臂数量
    def arm_loop(self):
        """实现机器臂的 NSGA-II算法来优化机器臂数量"""
        v1, v2 = main_loop(pop_size, max_gen, machine_counts, self.arm)