from Config import *
from Map import *
from robot_arm import *
from car import *

#创建初始地图
# 创建一个二维空间
# 创建一个 10x10 的车间地图
map = MapSpace(map_rows, map_cols,map_fill_char,road_width,work_width_up,work_width_down,work_name_up,work_name_down)

# 打印车间地图
map.display()
#显示生产单元的机器臂分配
# 初始化车间对象
init_arm = Arm(work_name_up, work_name_down, unit_numbers_up, unit_numbers_down, total_machines, machine_power,complexity)
# 随机分配所有机器
init_arm.distribute_machines_randomly()
# 生成任务
tasks = init_arm.generate_tasks()
# 显示每个生产单元的能量消耗
init_arm.display_unit_energy(tasks)
# 计算整个车间的总能量消耗
total_energy = init_arm.calculate_total_energy(tasks)
print(f"\n整个车间的总能量消耗: {total_energy} J")

# 初始化车辆分配对象
car = VehicleDistribution(work_name_up, work_name_down, unit_numbers_up, unit_numbers_down, total_vehicles)
# 随机分配所有车辆
car.distribute_vehicles_randomly()
