from Config import *
import random


class VehicleDistribution:
    def __init__(self, work_name_up, work_name_down, unit_numbers_up, unit_numbers_down, total_vehicles):
        # 初始化生产区名称和单元数
        self.work_name_up = work_name_up
        self.work_name_down = work_name_down
        self.unit_numbers_up = unit_numbers_up
        self.unit_numbers_down = unit_numbers_down
        self.total_vehicles = total_vehicles  # 总车辆数量

        # 创建字典来存储每个生产区的车辆数
        self.vehicles_count = {}

        # 初始化生产区的车辆数
        self._initialize_cells()

    def _initialize_cells(self):
        """初始化生产区及其对应的车辆数"""
        # 上半部分生产区
        for zone, unit_count in zip(self.work_name_up, self.unit_numbers_up):
            self.vehicles_count[zone] = 0  # 每个生产区初始分配的车辆数为 0

        # 下半部分生产区
        for zone, unit_count in zip(self.work_name_down, self.unit_numbers_down):
            self.vehicles_count[zone] = 0  # 为空单元初始化车辆数为 0

    def distribute_vehicles_randomly(self):
        """随机将车辆分配到生产区，并显示每个生产区分配的车辆数量"""
        available_zones = []  # 存储所有生产区

        # 先把所有生产区加入到 available_zones 中
        for zone in self.vehicles_count:
            available_zones.append(zone)  # 以生产区名称存储

        # 存储每个生产区分配到的车辆数量
        vehicles_per_zone = {zone: 0 for zone in self.vehicles_count}

        # 随机分配车辆
        for _ in range(self.total_vehicles):
            zone = random.choice(available_zones)
            self.vehicles_count[zone] += 1  # 增加该生产区的车辆数
            vehicles_per_zone[zone] += 1  # 增加该生产区的车辆总数
            #print(f"分配了 1 台车到 {zone}")

    def display_vehicle_count(self):
        """显示每个生产区和分配的车辆数"""
        print("\n当前生产区车辆分布：")
        for zone, vehicle_count in self.vehicles_count.items():
            print(f"{zone}: {vehicle_count} 台车")

