#map配置

map_rows=15#地图的宽
map_cols=39#地图的长
map_fill_char=' '

# ------------------------

#车间配置
road_width=1#道路宽度

#生产区墙壁位置
work_width_up=[11,17,30,map_cols-1]#上部墙壁
work_width_down=[15,33,map_cols-1]#下部墙壁

#生产区集合A
work_name_up=['组装区','铸造区','清洗区','包装区']
work_name_down=['焊接区','喷漆区','配置区']

# 每个生产区的单元数
unit_numbers_up = [3, 2, 3, 5]  # 对应 work_name_up 每个区的单元数
unit_numbers_down = [2, 4, 3]    # 对应 work_name_down 每个区的单元数
# 将生产区名称和单元数存储到字典中
cell_number={}
for zone, units in zip(work_name_up, unit_numbers_up):
    cell_number[zone] = units

for zone, units in zip(work_name_down, unit_numbers_down):
    cell_number[zone] = units

# 打印字典，查看每个生产区的单元数
#print(cell_number)

# ------------------------

#机器臂配置
total_machines = 30  # 总机器数
# ------------------------

#小车配置