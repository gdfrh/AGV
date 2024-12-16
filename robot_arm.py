from Config import *
import random


class Arm:
    def __init__(self, work_name_up, work_name_down, unit_numbers_up, unit_numbers_down, total_machines):
        # 初始化生产区名称和单元数
        self.work_name_up = work_name_up
        self.work_name_down = work_name_down
        self.unit_numbers_up = unit_numbers_up
        self.unit_numbers_down = unit_numbers_down
        self.total_machines = total_machines  # 总机器数量

        # 创建字典来存储每个生产单元的机器数
        self.machines_count = {}

        # 初始化生产区的机器数
        self._initialize_cells()

    def _initialize_cells(self):
        """初始化生产区及其对应的机器数"""
        # 上半部分生产区
        for zone, unit_count in zip(self.work_name_up, self.unit_numbers_up):
            self.machines_count[zone] = [0] * unit_count  # 为每个生产单元初始化机器数为 0

        # 下半部分生产区
        for zone, unit_count in zip(self.work_name_down, self.unit_numbers_down):
            self.machines_count[zone] = [0] * unit_count  # 为空单元初始化机器数为 0

    def distribute_machines_randomly(self):
        """随机将机器分配到生产单元，并显示每个生产区分配的机器数量"""
        available_units = []  # 存储所有生产单元的可用位置（生产区 + 单元索引）

        # 先把所有单元加入到 available_units 中
        for zone in self.machines_count:
            for i in range(len(self.machines_count[zone])):
                available_units.append((zone, i))  # 以 (生产区, 单元索引) 存储每个生产单元

        # 存储每个生产区分配到的机器数量
        machines_per_zone = {zone: 0 for zone in self.machines_count}

        # 随机分配机器
        for _ in range(self.total_machines):
            zone, unit_index = random.choice(available_units)
            self.machines_count[zone][unit_index] += 1  # 增加该单元的机器数
            machines_per_zone[zone] += 1  # 增加该生产区的机器总数
            #print(f"分配了 1 台机器到 {zone} 的单元 {unit_index + 1}")

        # 输出每个生产区分配到的机器数量
        print("\n每个生产区分配的机器数量：")
        for zone, count in machines_per_zone.items():
            print(f"{zone}: {count} 台机器")

    def display_machine_count(self):
        """显示每个生产区中各单元的机器数"""
        for zone, units in self.machines_count.items():
            print(f"{zone}: {', '.join([str(unit) for unit in units])} 台机器")

