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
unit_numbers_up = [2, 2, 1, 3]  # 对应 work_name_up 每个区的单元数
unit_numbers_down = [2, 1, 3]    # 对应 work_name_down 每个区的单元数
# 将生产区名称和单元数存储到字典中
cell_number={}
for zone, units in zip(work_name_up, unit_numbers_up):
    cell_number[zone] = units

for zone, units in zip(work_name_down, unit_numbers_down):
    cell_number[zone] = units

# 打印字典，查看每个生产区的单元数
#print(cell_number)
"""
我们进行假设，较为简单的生产区的生产单元增加机器臂后所需要的时间会显著下降
较为复杂的生产区的生产单元增加机器臂后所需要的工作时间会下降，但是下降幅度较小
暂定功率保持不变
"""
complexity = {
    '组装区': 'simple',
    '铸造区': 'complex',
    '清洗区': 'simple',
    '包装区': 'complex',
    '焊接区': 'complex',
    '喷漆区': 'simple',
    '配置区': 'complex'
}
# ------------------------
#机器臂配置
total_machines = 30  # 总机器数
machine_power = 100  # 每台机器的功率（单位：W）
work_time=1          #运行时间,暂未使用

# ------------------------
#小车配置
total_vehicles = 15  # 总车辆数