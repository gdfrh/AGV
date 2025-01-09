from Config import *
import random
from order import Order
from order import OrderManager


class Car:
    def __init__(self, work_name, unit_numbers, total_vehicles):
        # 初始化生产区名称和单元数
        self.work_name = work_name
        self.unit_numbers = unit_numbers
        self.total_vehicles = total_vehicles  # 总车辆数量


        # 创建字典来存储每个生产区的车辆数
        self.vehicles_count = {}

        # 初始化生产区的车辆数
        self._initialize_cells()

    def _initialize_cells(self):
        """初始化生产区及其对应的车辆数"""
        for zone, unit_count in zip(self.work_name, self.unit_numbers):
            self.vehicles_count[zone] = 0  # 每个生产区初始分配的车辆数为 0


    def distribute_vehicles_randomly(self):
        """首先为每个生产区分配一个车辆，然后随机将剩余车辆分配到生产区"""
        available_zones = []  # 存储所有生产区

        # 1. 为每个生产区分配一个车辆
        for zone in self.vehicles_count:
            self.vehicles_count[zone] = 1  # 每个生产区至少分配一个车辆
            available_zones.append(zone)  # 存储生产区名称

        # 剩余车辆数量
        remaining_vehicles = self.total_vehicles - len(self.vehicles_count)

        # 2. 随机分配剩余的车辆
        while remaining_vehicles > 0:
            zone = random.choice(available_zones)
            self.vehicles_count[zone] += 1  # 为该生产区分配一辆车
            remaining_vehicles -= 1  # 剩余车辆数减少1

        # 输出每个生产区分配到的车辆数量
        # self.display_vehicle_count()

    # def display_vehicle_count(self):
    #     """显示每个生产区和分配的车辆数"""
    #     print("\n当前生产区车辆分布：")
    #     for zone, vehicle_count in self.vehicles_count.items():
    #         print(f"{zone}: {vehicle_count} 台车")
