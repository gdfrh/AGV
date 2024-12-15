from Config import *
from Map import *


#创建初始地图
# 创建一个二维空间
# 创建一个 10x10 的车间地图
workshop = MapSpace(map_rows, map_cols,map_fill_char,road_width,work_width_up,work_width_down,work_name_up,work_name_down)

# 打印车间地图
workshop.display()


