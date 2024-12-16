from Config import *
from Map import *
from robot_arm import *

#创建初始地图
# 创建一个二维空间
# 创建一个 10x10 的车间地图
map = MapSpace(map_rows, map_cols,map_fill_char,road_width,work_width_up,work_width_down,work_name_up,work_name_down)

# 打印车间地图
map.display()
#显示生产单元的机器臂分配
init_arm = Arm(work_name_up, work_name_down, unit_numbers_up, unit_numbers_down, total_machines)
# 随机分配所有机器
init_arm.distribute_machines_randomly()
init_arm.display_machine_count()


