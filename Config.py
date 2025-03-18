# map配置

map_rows = 15  # 地图的宽
map_cols = 39  # 地图的长
map_fill_char = ' '

# ------------------------
# 车间配置
road_width = 1  # 道路宽度

# 生产区墙壁位置
work_width_up = [11, 17, 30, map_cols-1]  # 上部墙壁
work_width_down = [15, 33, map_cols-1]  # 下部墙壁

# 生产区集合A
work_name_up = ['组装区', '铸造区', '清洗区', '包装区']
work_name_down = ['焊接区', '喷漆区', '配置区']
work_name = work_name_up + work_name_down
work_name_order = {  # 用来处理订单对应输出
    0: "组装区",
    1: "铸造区",
    2: "清洗区",
    3: "包装区",
    4: "焊接区",
    5: "喷漆区",
    6: "配置区"
}
"""生产区处理时间"""
# 我想试一试固定处理时间，即订单经过某生产区的停留时间
processing_time = {
'run_time' : 30,  # 运行时间（秒），
'run_power' : 33.70,  # 运行功率（瓦特），
'sleep_time' : 15,  # 休眠时间（秒），
'sleep_power' : 19.1  # 休眠功率（瓦特），
}

# 生产区距离矩阵
# 假设这是一个已知的距离矩阵（单位：米）
distance_matrix = [
    [0,  10, 20, 30, 40, 50, 60, 70],  # 组装区 -> [组装区, 铸造区, 清洗区, ...,离开车间]
    [10,  0, 15, 25, 35, 45, 55, 50],  # 铸造区 -> [组装区, 铸造区, 清洗区, ...,离开车间]
    [20, 15,  0, 10, 20, 30, 40, 60],  # 清洗区 -> [组装区, 铸造区, 清洗区, ...,离开车间]
    [30, 25, 10,  0, 10, 20, 30, 30],  # 包装区 -> [组装区, 铸造区, 清洗区, ...,离开车间]
    [40, 35, 20, 10,  0, 10, 20, 20],  # 焊接区 -> [组装区, 铸造区, 清洗区, ...,离开车间]
    [50, 45, 30, 20, 10,  0, 10, 90],  # 喷漆区 -> [组装区, 铸造区, 清洗区, ...,离开车间]
    [60, 55, 40, 30, 20, 10,  0, 10],  # 配置区 -> [组装区, 铸造区, 清洗区, ...,离开车间]
    [70, 50, 60, 30, 20, 90, 10,  0],  # 离开区 -> [组装区, 铸造区, 清洗区, ...,离开车间]
]

# # 每个生产区的单元数
# unit_numbers_up = [2, 2, 1, 3]  # 对应 work_name_up 每个区的单元数
# unit_numbers_down = [2, 1, 3]    # 对应 work_name_down 每个区的单元数
# unit_numbers = unit_numbers_up + unit_numbers_down
# # 将生产区名称和单元数存储到字典中
# cell_number = {}
# for zone, units in zip(work_name, unit_numbers):
#     cell_number[zone] = units

"""
我们进行假设，较为简单的生产区的生产单元增加机器臂后所需要的时间会显著下降
较为复杂的生产区的生产单元增加机器臂后所需要的工作时间会下降，但是下降幅度较小
暂定功率保持不变
修改一下吧，为了保证样本丰富性，我为每个生产区设定参数
"""
reduction_factor_time_list = [0.32, 0.37, 0.51, 0.63, 0.17, 0.33, 0.15]
reduction_factor_power_list = [0.26, 0.21, 0.47, 0.71, 0.11, 0.31, 0.15]

# ------------------------
# 机器臂配置
total_machines = 50  # 总机器数
machine_power = 30  # 每台机器的功率（单位：W）

# 存储机器臂的分配
machine_count_list = []  # 列表
machine_counts = []  # 解的机器臂分配
energy_counts = []  # 解的能量消耗
time_counts = []  # 解的时间消耗
# 生产区机器臂的生产单元最低所需的数量最低要求
zone_requirements = [
    ("组装区", 5),  # 组装区生产单元需要2个机器臂
    ("铸造区", 5),  # 铸造区生产单元需要2个机器臂
    ("清洗区", 4),  # 清洗区生产单元需要1个机器臂
    ("包装区", 4),  # 包装区生产单元需要3个机器臂
    ("焊接区", 3),  # 焊接区生产单元需要2个机器臂
    ("喷漆区", 3),  # 喷漆区生产单元需要1个机器臂
    ("配置区", 2)  # 配置区生产单元需要3个机器臂
]
# ------------------------
# 小车配置
total_agv = 15  # 总车辆数
agv_speed = 5000    # 小车速度

# ------------------------
# NSGA2参数
pop_size = 50 # 每一代种群数量
max_gen = 300   # 最高代数
number_limits = 0.1  # 交叉变异对象的数量需求
mutation_probability = 0.2  # 变异概率
# ------------------------
# 订单数
num_orders = 10
# 订单相似破坏率
similarity_percent = 0.2
# ALNS迭代
iterations = 0

