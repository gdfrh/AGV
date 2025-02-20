from Map import *
from Config import *
from schedule import Schedule


# 创建初始地图
# 创建一个二维空间# 创建一个 10x10 的车间地图
# map = MapSpace(map_rows, map_cols, map_fill_char, road_width, work_width_up, work_width_down, work_name_up, work_name_down)
# # 打印车间地图
# map.display()
scheduler = Schedule(work_name, total_machines, machine_power, num_orders, zone_requirements)
scheduler.arm_random()
scheduler.arm_loop()





