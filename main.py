from Config import *
from Map import *
from robot_arm import *
from car import *
from NSGA2 import *

#创建初始地图
# 创建一个二维空间# 创建一个 10x10 的车间地图
map = MapSpace(map_rows, map_cols,map_fill_char,road_width,work_width_up,work_width_down,work_name_up,work_name_down)
# 打印车间地图
map.display()

#显示生产单元的机器臂分配
# 初始化车间对象
init_arm = Arm(work_name_up, work_name_down, unit_numbers_up, unit_numbers_down, total_machines, machine_power,complexity)
# 随机分配所有机器
init_arm.distribute_machines_randomly()
# 生成任务
init_arm.calculate_and_display_energy(init_arm)


# 初始化车辆分配对象
car = VehicleDistribution(work_name_up, work_name_down, unit_numbers_up, unit_numbers_down, total_vehicles)
# 随机分配所有车辆
car.distribute_vehicles_randomly()


#v1, v2 = main_loop(pop_size, max_gen, generate_initial_population(pop_size))

"""
我想循环随机分配机器臂，将每次随机分配的结果作为解，function分别得出这种分配求得的能耗和时间，现在的问题是如何表示这个解，用编码？字典？
我现在调用任务是随机的，是否应该用固定的任务来查看配置的优先级
"""