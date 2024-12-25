from Config import *
from Map import *
from robot_arm import *
from car import *
from NSGA2 import *
import copy

#创建初始地图
# 创建一个二维空间# 创建一个 10x10 的车间地图
map = MapSpace(map_rows, map_cols,map_fill_char,road_width,work_width_up,work_width_down,work_name_up,work_name_down)
# 打印车间地图
map.display()

#显示生产单元的机器臂分配
# 初始化车间对象
init_arm = Arm(work_name_up, work_name_down, unit_numbers_up, unit_numbers_down, total_machines, machine_power,complexity)


# 随机分配所有机器C
for _ in range(pop_size):
    init_arm.distribute_machines_randomly()
    new_list = copy.deepcopy(init_arm.display_machine_count())
    energy_count, time_count = init_arm.function_1(new_list)

    # 保留两位小数
    energy_count = round(energy_count, 2)
    time_count = round(time_count, 2)

    energy_counts.append(energy_count)
    time_counts.append(time_count)

    machine_counts.append(new_list)#记录机器臂数量



# 初始化车辆分配对象
car = VehicleDistribution(work_name_up, work_name_down, unit_numbers_up, unit_numbers_down, total_vehicles)
# 随机分配所有车辆
car.distribute_vehicles_randomly()


v1, v2 = main_loop(pop_size, max_gen,machine_counts ,init_arm)

"""
我想循环随机分配机器臂，将每次随机分配的结果作为解，function分别得出这种分配求得的能耗和时间，现在的问题是如何表示这个解，用编码？字典？
我现在调用任务是随机的，是否应该用固定的任务来查看配置的优先级
如果用编码的方式，那么比较容易进行最开始初始解以及第一代子代的构造，只需要变异（比如交换编码）然后提取到对应的生产单元，重新计算时间和能耗即可
字典不太好变异
"""