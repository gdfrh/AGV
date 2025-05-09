from Config import *
from order import *
import random
import time
import numpy as np
import math
import heapq
import itertools
import copy
from dispatch import Timeline


class Arm:
    def __init__(self, work_name, unit_numbers, total_machines, num_orders):
        # 初始化生产区名称和单元数
        self.work_name = work_name
        self.unit_numbers = unit_numbers  # 初始的生产单元个数
        self.unit_states = []  # 后续每个序列的生产单元个数
        self.total_machines = total_machines  # 总机器数量
        self.machines_count = {}  # 创建字典来存储每个生产单元的机器数
        # self.order_manager = OrderManager(work_name, num_orders)
        # self.orders = self.order_manager.get_orders()  # 调用 OrderManager来显示订单
        self.orders = [
            ['铸造区', '清洗区', '组装区', '配置区', '包装区'],
            ['铸造区', '清洗区', '组装区', '配置区', '包装区'],
            ['铸造区', '清洗区', '组装区', '配置区', '包装区'],
            ['铸造区', '清洗区', '组装区', '配置区', '包装区'],
            ['铸造区', '清洗区', '组装区', '配置区', '包装区'],
            ['铸造区', '清洗区', '焊接区', '配置区', '喷漆区', '包装区'],
            ['铸造区', '清洗区', '焊接区', '配置区', '喷漆区', '包装区'],
            ['铸造区', '清洗区', '焊接区', '配置区', '喷漆区', '包装区'],
            ['铸造区', '清洗区', '焊接区', '配置区', '喷漆区', '包装区'],
            ['铸造区', '清洗区', '焊接区', '配置区', '喷漆区', '包装区'],
            ['铸造区', '清洗区', '配置区', '包装区'],
            ['铸造区', '清洗区', '配置区', '包装区'],
            ['铸造区', '清洗区', '配置区', '包装区'],
            ['铸造区', '清洗区', '配置区', '包装区'],
            ['铸造区', '清洗区', '配置区', '包装区'],
]
        self.orders_list = []   # 用来记录每一个解的订单顺序排列
        """用False表示空闲，True表示忙碌"""
        self.work_status = {}  # 用来判断生产区的生产单元是否在工作
        self.start_time = {}  # 用来记录生产区工作的开始时间
        self.end_time = {}  # 用来记录生产区工作的结束时间
        self.timeline_record = {}  # 用来记录ALNS迭代中的最优值的时间节点
        self.timeline_history = []    # 存储时间轴的处理历史状态
        self.timeline_history_1 = []  # 存储时间轴的-1历史状态
        self.timeline_history_2 = []  # 存储时间轴的-2历史状态
        self.timeline_history_3 = []  # 存储时间轴的-3历史状态
        self.agv_timeline_history = []  # 存储小车时间轴的历史状态
        self.iteration_value = []   # 记录迭代效果
        self.weight = []  # 记录权重
        self.counter = []  # 记录算子使用次数
        self.SA_temperature = [init_temperature] * (2 * pop_size + 1)  # 记录模拟退火温度
        # ------------------------
        """小车"""
        self.agv_count = []  # 创建列表来存储每个生产区的小车数
        self.agv_states = {}  # 用来判断生产区的小车是否在工作
        self.agv_start_time = {}  # 用来记录生产区小车工作的开始时间
        self.agv_end_time = {}  # 用来记录生产区小车工作的结束时间
        # ------------------------
        self._initialize_cells()  # 初始化生产区的机器数

    def _initialize_cells(self):
        """初始化生产区及其对应的机器数"""
        for zone, unit_count in zip(self.work_name, self.unit_numbers):
            self.machines_count[zone] = [0] * unit_count  # 为每个生产单元初始化机器数为 0
            self.work_status[zone] = [False] * unit_count  # 初始时，生产区是空闲的
            self.start_time[zone] = [0] * unit_count  # 初始时，没有开始时间
            self.end_time[zone] = [0] * unit_count  # 初始时，没有结束时间
            self.agv_states[zone] = [False]  # 初始时，生产区的小车是空闲的

    def count_nonzero_machines(self):
        """统计每个生产区中机器臂不为0的生产单元数量"""
        non_zero_machines_count = {}  # 用来存储每个生产区的非零机器臂数量

        for zone in self.machines_count:
            # 统计当前生产区中机器臂不为0的生产单元数量
            if zone != work_name[-1]:
                non_zero_count = sum(1 for machine in self.machines_count[zone] if machine != 0)
                non_zero_machines_count[zone] = non_zero_count

        return non_zero_machines_count

    def distribute_machines_randomly(self):
        # 先初始化机器臂的数量
        self._initialize_cells()
        """随机将机器分配到生产单元，并显示每个生产区分配的机器数量"""
        available_units = []  # 存储所有生产单元的可用位置（生产区 + 单元索引）
        # 1. 根据需求给每个生产区分配机器
        remaining_machines = self.total_machines  # 总机器数量
        for zone, min_machines in zone_requirements:  # zone_requirements 中保存每个生产区需要的最小机器数量
            # 获取该生产区的单元数
            units = len(self.machines_count[zone])
            obj_units = random.randint(0, units - 1)
            # 先给每个生产单元分配最低机器臂数量
            self.machines_count[zone][obj_units] = min_machines
            remaining_machines -= min_machines
        # 2. 随机分配机器
        """如果我按最低部署，那么感觉很难找到种群，那我直接都可以按1个机器臂部署，变异再按最小数量？（未改）"""
        while remaining_machines > 0:
            zone = random.choice(self.work_name)  # 随机选择一个生产区
            unit_index = random.randint(0, len(self.machines_count[zone]) - 1)  # 随机选择一个生产单元
            # 分配每个生产区需要的机器数，而不是逐个机器分配
            """如果机器臂数量为0，则分配最低要求数"""
            if self.machines_count[zone][unit_index] == 0:
                min_required_machines = zone_requirements[self.work_name.index(zone)][1]  # 获取该生产区的最小机器需求
                if remaining_machines >= min_required_machines:
                    self.machines_count[zone][unit_index] = min_required_machines
                    remaining_machines -= min_required_machines
                """如果已分配机器臂，就多分配1个"""
            elif self.machines_count[zone][unit_index] != 0:
                self.machines_count[zone][unit_index] += 1
                remaining_machines -= 1

        """根据生产区机器臂分配比例来分配小车数量"""
        # 1. 计算每个生产区机器臂不为零的生产单元数量
        non_zero_machines_count = self.count_nonzero_machines()

        # 2. 计算所有生产区机器臂不为零的生产单元总和
        total_non_zero = sum(non_zero_machines_count.values())

        # 3. 根据比例来分配小车数量
        agv_distribute = [1] * len(work_name[:-1])  # 每个生产区分配1个小车
        remaining_agv = total_agv - sum(agv_distribute)  # 总小车数量减去已经分配的数量
        if remaining_agv > 0:
            # 计算每个生产区应该分配的比例
            for i, (zone, non_zero_count) in enumerate(non_zero_machines_count.items()):
                proportion = non_zero_count / total_non_zero
                car_count_for_zone = round(proportion * remaining_agv)  # 根据比例计算剩余的小车数量
                if car_count_for_zone <= remaining_agv:
                    agv_distribute[i] += car_count_for_zone
                    remaining_agv -= car_count_for_zone
        # 确保分配的小车总数为 total_agv
        total_distributed_agv = sum(agv_distribute)
        if total_distributed_agv != total_agv:
            # 计算差额
            difference = total_agv - total_distributed_agv
            # 找到小车最多的生产区
            max_agv_zone = agv_distribute.index(max(agv_distribute))
            # 将差额加到小车最多的生产区
            agv_distribute[max_agv_zone] += difference
        # agv_distribute = [2,6,1,2,2,2]
        # 4. 对每个生产区的生产单元按机器数量从大到小排序
        for zone in self.machines_count:
            # 排序每个生产区的单元，按机器数量从大到小
            self.machines_count[zone] = sorted(self.machines_count[zone], reverse=True)

        return agv_distribute

    def display_machine_count(self):
        """返回每个生产区中各单元的机器数，并存储到列表中"""
        machine_count_list = []  # 列表
        for zone, units in self.machines_count.items():
            # 将每个生产区的机器数存储到字典中
            machine_count_list += units  # 每个单元的机器数
        # print(machine_count_list)
        merged_str = ''.join(map(str, machine_count_list))
        # 将连接后的字符串转换为整数
        merged_number = int(merged_str)
        machine_count_list_renew = merged_number
        return machine_count_list_renew

    def calculate_reduction(self, zone_name, machine_count):
        # 求解矩阵行列
        row_idx = work_name.index(zone_name)
        col_idx = machine_count - 1
        cope_time = processing_time_matrix[row_idx][col_idx]
        cope_energy = energy_consumption_matrix[row_idx][col_idx]

        return cope_time, cope_energy

    def calculate_transport_time(self, start_zone, end_zone, distance_matrix, work_name_order, speed):
        """由于传递过来的订单是汉字，我们反映射为数字"""
        # Step 1: 创建一个反向字典，将生产区汉字名称映射到数字索引
        zone_to_index = {name: index for index, name in work_name_order.items()}

        # Step 2: 使用反向字典，将汉字生产区转换为数字索引
        start_index = zone_to_index[start_zone]
        end_index = zone_to_index[end_zone]

        # Step 3: 使用转换后的数字索引从距离矩阵中获取距离
        distance = distance_matrix[start_index][end_index]

        # Step 4: 计算运输时间（假设速度是已知的，单位为距离/时间）
        transport_time = distance / speed  # 时间 = 距离 / 速度
        return transport_time

    def find_false_max_machines(self, zone):
        """找出某个生产区中状态为 False 的生产单元中，机器臂最多的单元"""
        max_machines = 0
        max_unit_index = -1  # 记录机器臂最多的生产单元的索引

        for i, status in enumerate(self.work_status[zone]):
            if not status:  # 如果当前生产单元处于空闲状态
                if self.machines_count[zone][i] > max_machines:
                    max_machines = self.machines_count[zone][i]
                    max_unit_index = i

        # 返回机器臂最多的生产单元及其机器臂数量
        if max_unit_index != -1:
            return max_unit_index, max_machines
        else:
            return None  # 如果没有空闲单元

    def find_min_wait_time(self, zone, obj_type):
        """找出某个生产区中所需等待时间最短的单元或小车
        :param zone: 生产区域
        :param obj_type: 'unit' 表示生产单元，'agv' 表示小车
        """
        min_wait_time = float('inf')
        min_unit_index = -1  # 记录等待时间最短的生产单元的索引

        # 获取当前时间一次，避免在循环中多次调用
        current_time = time.time()

        if obj_type == 'unit':
            status_list = self.work_status[zone]
            end_time_list = self.end_time[zone]
        elif obj_type == 'agv':
            status_list = self.agv_states[zone]
            end_time_list = self.agv_end_time[zone]
        else:
            raise ValueError("Type must be 'unit' or 'agv'")

        for i, status in enumerate(status_list):
            if status:  # 如果当前单元或小车处于忙碌状态
                remaining_time = end_time_list[i] - current_time
                if remaining_time < min_wait_time:
                    min_wait_time = remaining_time
                    min_unit_index = i

        # 返回等待时间最短的单元索引及其等待时间
        if min_unit_index != -1:
            return min_unit_index, min_wait_time
        else:
            return None  # 如果没有空闲单元

    def find_random_idle_agv(self, start_zone, agv_list):
        """随机找到一个空闲的小车，并返回其索引"""
        # 获取当前生产区的工作状态
        agv_work_status = self.agv_states[start_zone]
        # 找到所有空闲的小车的索引（状态为 False）
        idle_cars = [index for index, status in enumerate(agv_work_status) if not status]
        idx = random.choice(idle_cars)
        # 计算该小车在所有小车列表中的全局索引
        zone_idx = work_name.index(start_zone)
        global_agv_index = sum(agv_list[:zone_idx]) + idx
        return global_agv_index

    def find_last_none_index(self, matrix, idx):
        # 获取指定行
        row = matrix[idx]
        # 从后向前查找 None 的索引
        for i in range(len(row) - 1, -1, -1):
            if row[i] is None:
                return i
        return -1  # 如果没有找到 None，则返回 -1

    def find_index_of_agv(self, agv_count, idx):
        # 根据全局小车索引找生产区小车索引
        car_count = 0
        # 遍历生产区
        for zone_index, cars in enumerate(agv_count):
            # 检查当前生产区是否包含目标索引
            if idx < car_count + cars:
                # 找到目标生产区
                production_zone = zone_index  # 生产区从0开始
                car_in_zone = idx - car_count  # 生产区内的小车编号，从0开始
                break
            # 更新已遍历的小车数量
            car_count += cars
        return production_zone, car_in_zone

    def order_time_and_power(self, orders, step_idx, type):
        """计算一个订单的处理时间和功率消耗，包括生产区时间和运输时间"""
        order_time = 0  # 订单步骤完成所需时间
        order_power = 0  # 订单步骤完成所需功率
        use_time = [0] * num_orders  # 每个订单所用时间的列表
        use_power = 0  # 总共消耗的功率

        timeline_history = []  # 存储时间轴的处理历史状态
        timeline_history_1 = []  # 存储时间轴的-1历史状态
        timeline_history_2 = []  # 存储时间轴的-2历史状态
        timeline_history_3 = []  # 存储时间轴的-3历史状态
        agv_timeline_history = []  # 存储小车时间轴的历史状态

        """遍历所有生产区和单元，将机器臂数量为 0 的生产单元的状态改为忙碌，避免出现选择0个机器臂的生产单元进行操作"""
        for zone in self.machines_count:
            for i in range(len(self.machines_count[zone])):
                if self.machines_count[zone][i] == 0:  # 如果该生产单元的机器臂数量为 0
                    self.work_status[zone][i] = True  # 将该生产单元状态设为忙碌
                    self.end_time[zone][i] = float('inf')  # 设置一个很大的结束时间，确保不会被选中

        time_line = Timeline()  # 引入时间轴

        # 找到最长的订单长度
        max_steps = max(len(order) for order in orders)

        # 创建矩阵，并填充 0 占位
        order_matrix = np.full((len(orders), max_steps), 0, dtype=object)
        # 获取矩阵的维度
        num_rows, num_cols = order_matrix.shape
        # 填充订单数据
        for i, order in enumerate(orders):
            order_matrix[i, :len(order)] = order  # 只填充有效的订单步骤

        # 初始化条件或一个标志来确保至少进入循环一次
        first_iteration = True
        idx = []  # 记录时间节点的索引
        point_type = 'start'  # 初始化断点类型
        while first_iteration or time_line.timeline:

            if all(point == float('inf') for point in time_line.timeline):
                if type == 'picture':
                    self.timeline_record['timeline_history'] = timeline_history
                    self.timeline_record['timeline_history_1'] = timeline_history_1
                    self.timeline_record['timeline_history_2'] = timeline_history_2
                    self.timeline_record['timeline_history_3'] = timeline_history_3
                    self.timeline_record['agv_timeline_history'] = agv_timeline_history

                break  # 跳出循环，停止处理，或者根据需要进行其他操作
            # 记录每一行是否已经找到了非 None 值，每一次都需要重置
            if point_type is None and idx is None:
                print('死锁')
                print(self.work_status)
                print(time_line.timeline)
                print(time_line.agv_timeline)
                break

            row_found = [False] * num_rows
            if first_iteration:
                first_iteration = False

            timeline_one = [0] * num_rows
            timeline_two = [0] * num_rows
            if idx:  # 不是第一次迭代，idx列表不为空，存在时间节点最小值，即完成了操作
                if point_type == 'order':  # 是订单断点，此时订单结束了生产单元的操作
                    # 记录小车开始工作时间，方便绘制甘特图
                    agv_timeline_one = [0] * total_agv
                    agv_timeline_two = [0] * total_agv
                    agv_order = [0] * total_agv

                    for index in idx:
                        zone_idx = self.find_last_none_index(order_matrix, index)  # 找到最后一个None的索引位置，即是对应生产区的位置
                        obj_zone = orders[index][zone_idx]  # 目标生产区
                        if zone_idx == num_cols - 1 or (
                                zone_idx + 1 < num_cols and order_matrix[index, zone_idx + 1] == 0):
                            # 如果None为订单最后一步或者None后面是0，即完成了该订单,记录时间,并将时间节点置为无穷大
                            use_time[index] = time_line.timeline[index]
                            time_line.timeline[index] = float('inf')
                            self.work_status[obj_zone][time_line.step[index]] = False  # 生产单元变为空闲
                        else:  # 不是最后一个生产区
                            """现在的情况就是有车就进，就不会出现存在车时新的order断点出现，还要遍历寻找的情况"""
                            if False in self.agv_states[obj_zone]:  # 存在空闲小车
                                self.work_status[obj_zone][time_line.step[index]] = False  # 生产单元变为空闲
                                agv_idx = self.find_random_idle_agv(obj_zone, self.agv_count[step_idx])  # 找到小车全局索引
                                _, idx_of_states = self.find_index_of_agv(self.agv_count[step_idx], agv_idx)
                                self.agv_states[obj_zone][idx_of_states] = True  # 占据空闲小车
                                # 这一个订单完成了操作，这一个订单在该断点不用操作了
                                row_found[index] = True
                                transport_time = self.calculate_transport_time(obj_zone, orders[index][zone_idx + 1],
                                                                               distance_matrix, work_name_order,
                                                                               agv_speed)
                                """我们在小车时间节点储存元组，（目标时间，目的地，小车的操作{-1是返回，1是送往},订单）"""
                                time_line.agv_timeline[agv_idx] = (
                                time_line.current_time + transport_time, orders[index][zone_idx + 1], 1, index)
                                # 此时要将时间轴的时间改为忙碌:-1
                                time_line.timeline[index] = -1
                                # 添加开始的时间点，便于画图
                                timeline_one[index] = time_line.current_time
                                timeline_two[index] = time_line.current_time + transport_time

                                agv_timeline_one[agv_idx] = time_line.current_time
                                if index == 0:
                                    agv_order[agv_idx] = -1
                                else:
                                    agv_order[agv_idx] = index
                                agv_timeline_two[agv_idx] = time_line.current_time + 2 * transport_time
                            elif False not in self.agv_states[obj_zone]:  # 不存在空闲小车:-2
                                time_line.timeline[index] = -2

                                timeline_one[index] = time_line.current_time
                            # -1代表后面的两个是运输的开始和结束，-2只代表下一个是等待小车的开始
                    timeline_history_1.append(time_line.timeline[:])
                    timeline_history_1.append(timeline_one[:])
                    timeline_history_1.append(timeline_two[:])

                    timeline_history_2.append(time_line.timeline[:])
                    timeline_history_2.append(timeline_one[:])
                    # 有空闲小车时会记录小车出发的时间，time_line.agv_timeline[:]会记录小车送到的时间
                    agv_timeline_history.append(agv_timeline_one[:])
                    agv_timeline_history.append(agv_order[:])
                    agv_timeline_history.append(agv_timeline_two[:])

                    """这里释放了生产单元的后续是是否有人需要生产单元√
                    不存在小车的后续是在其他断点出现小车空闲时，需要有判断找车的地方"""
                if point_type == 'agv':  # 是小车断点，此时要区分是哪一种断点，送到还是返回
                    # 记录小车开始工作时间，方便绘制甘特图
                    for index in idx:
                        if time_line.agv_timeline[index][2] == 1:
                            # 如果小车送达了订单，先读取是哪一个订单，将order时间节点变为-3送达生产区，表示空闲，并添加小车返回断点
                            order_idx = time_line.agv_timeline[index][3]
                            # 获取目标生产单元，添加返回断点
                            obj_zone = time_line.agv_timeline[index][1]
                            obj_zone_renew = orders[order_idx][(orders[order_idx].index(obj_zone) - 1)]

                            transport_time = self.calculate_transport_time(obj_zone, obj_zone_renew, distance_matrix,
                                                                           work_name_order, agv_speed)
                            """我们在小车时间节点储存元组，（目标时间，目的地，小车的操作{-1是返回，1是送往},订单）"""
                            time_line.agv_timeline[index] = (
                                time_line.current_time + transport_time, obj_zone_renew, -1, order_idx)

                            time_line.timeline[order_idx] = -3
                            time_line.agv_step[order_idx] = index

                            timeline_one[order_idx] = time_line.current_time

                            timeline_history_3.append(timeline_one[:])
                            """至于这里断点的操作会在下面的矩阵里面进行，这里不能给小车断点，不知道有没有空闲生产单元
                            这里的后续是是否存在空闲生产单元与之相关的处理,按照矩阵遍历顺序对所有处于这一状态的判断是否空闲生产区None_unit"""

                        elif time_line.agv_timeline[index][2] == -1:
                            # 如果小车返回了出发点，将agv时间节点变为None，表示空闲,并且将states变为False
                            obj_zone = time_line.agv_timeline[index][1]  # 找到目的地，即出发地
                            # 找到了是生产区的第几辆车
                            zone_idx, idx_of_states = self.find_index_of_agv(self.agv_count[step_idx], index)
                            self.agv_states[obj_zone][idx_of_states] = False
                            # 元组不能直接修改
                            # time_line.agv_timeline[index][0] = None
                            # 将元组转换为列表
                            temp_list = list(time_line.agv_timeline[index])
                            # 修改元组
                            for i in range(len(temp_list)):
                                temp_list[i] = None
                            # 将列表转换回元组
                            time_line.agv_timeline[index] = tuple(temp_list)
                            """这里生产区小车空闲，但是还没设置在哪里处理小车寻找
                            这里的后续是是否有空闲的订单需要找车与之相关的处理，对矩阵遍历，对所有的相同状态的需要小车的订单寻找小车"""

            agv_timeline_one = [0] * total_agv
            agv_timeline_two = [0] * total_agv
            agv_order = [0] * total_agv
            for row in range(num_rows):
                for col in range(num_cols):
                    timeline_one = [0] * num_rows
                    timeline_two = [0] * num_rows
                    timeline_three = [0] * num_rows
                    timeline_four = [0] * num_rows
                    # 检查当前行是否已经找到非 None 0 值，即每一个非None值可以进行一个节点的添加，如果已经找到就不需要管该订单后续部分
                    if not row_found[row] and order_matrix[row, col] is not None and order_matrix[row, col] != 0:
                        # 找到目标生产区
                        start_zone = orders[row][col]
                        # 标记此行已找到非 None 值
                        row_found[row] = True
                        if point_type == 'order' or point_type == 'start' or point_type == 'agv':
                            # 存在空闲生产单元并且订单位于小车之上并且完成了自己的工作处于等待时间
                            # -3代表小车已送达订单但是没有生产单元
                            if False in self.work_status[start_zone] and (time_line.timeline[row] == -3):

                                # 所找到的生产区变为None
                                order_matrix[row, col] = None
                                max_unit_index, max_machines = self.find_false_max_machines(start_zone)  # 暂时选择机器臂最多的单元
                                order_time, order_total_power = self.calculate_reduction(start_zone, max_machines)
                                order_power = order_total_power * order_time * random.uniform(0.95,1.00)
                                # 记录使用时间和功率
                                use_power += order_power
                                time_line.add_timeline((order_time + time_line.current_time), row,
                                                       max_unit_index)  # 添加订单时间节点，可以知道对应的订单
                                self.work_status[start_zone][max_unit_index] = True  # 占据空闲生产单元
                                timeline_one[row] = time_line.current_time
                                timeline_two[row] = order_time + time_line.current_time
                                timeline_history.append(timeline_one[:])
                                timeline_history.append((start_zone,max_unit_index))
                                timeline_history.append(timeline_two[:])

                                if point_type == 'start':
                                    timeline_history_3.append([0] * num_rows)
                                    timeline_history_3.append(timeline_one[:])
                                if point_type != 'start':
                                    timeline_history_3.append(timeline_one[:])

                        if point_type == 'agv':
                            # 如果是小车断点，得判断出现了哪些情况
                            # 现在由于生产区小车开始空闲，需要寻找是否有地方需要小车，即寻找存在order节点为-2:
                            # 现在找到的是None后面的一个，所以对前一个生产区判断小车
                            timeline_one = [0] * num_rows
                            timeline_two = [0] * num_rows
                            if False in self.agv_states[orders[row][col - 1]] and (time_line.timeline[row] == -2):
                                self.work_status[orders[row][col - 1]][time_line.step[row]] = False  # 生产单元变为空闲
                                agv_idx = self.find_random_idle_agv(orders[row][col - 1],
                                                                    self.agv_count[step_idx])  # 找到小车全局索引
                                _, idx_of_states = self.find_index_of_agv(self.agv_count[step_idx], agv_idx)
                                self.agv_states[orders[row][col - 1]][idx_of_states] = True  # 占据空闲小车
                                transport_time = self.calculate_transport_time(start_zone, orders[row][col - 1],
                                                                               distance_matrix, work_name_order,
                                                                               agv_speed)
                                """我们在小车时间节点储存元组，（目标时间，目的地，小车的操作{-1是返回，1是送往},订单）"""
                                time_line.agv_timeline[agv_idx] = (
                                    time_line.current_time + transport_time, start_zone, 1, row)
                                # 此时要将时间轴的时间改为忙碌:-1
                                time_line.timeline[row] = -1
                                timeline_three[row] = time_line.current_time
                                timeline_four[row] = time_line.current_time + transport_time
                                timeline_history_1.append(time_line.timeline[:])
                                timeline_history_1.append(timeline_three[:])
                                timeline_history_1.append(timeline_four[:])

                                timeline_history_2.append(timeline_three[:])
                                # 记录小车时间
                                agv_timeline_one[agv_idx] = time_line.current_time
                                if row == 0:
                                    agv_order[agv_idx] = -1
                                else:
                                    agv_order[agv_idx] = row
                                agv_timeline_two[agv_idx] = time_line.current_time + 2 * transport_time

                                # 如果订单被小车送走了，此时要判断是否存在小车在等
                                for rows in range(num_rows):
                                    for cols in range(num_cols):
                                        # 检查当前行是否已经找到非 None 0 值，即每一个非None值可以进行一个节点的添加，如果已经找到就不需要管该订单后续部分
                                        if not row_found[rows] and order_matrix[rows, cols] is not None and \
                                                order_matrix[rows, cols] != 0:
                                            # 找到目标生产区
                                            start_zone = orders[rows][cols]
                                            # 标记此行已找到非 None 值
                                            row_found[rows] = True
                                            timeline_one = [0] * num_rows
                                            timeline_two = [0] * num_rows
                                            if False in self.work_status[start_zone] and (
                                                    time_line.timeline[rows] == -3):
                                                # 所找到的生产区变为None
                                                order_matrix[rows, cols] = None
                                                max_unit_index, max_machines = self.find_false_max_machines(
                                                    start_zone)  # 暂时选择机器臂最多的单元
                                                order_time, order_total_power = self.calculate_reduction(start_zone, max_machines)
                                                # 生成一个0.99到1.01之间的随机浮动值
                                                order_power = order_total_power * order_time * random.uniform(0.95, 1.00)
                                                # 记录使用时间和功率
                                                use_power += order_power
                                                time_line.add_timeline((order_time + time_line.current_time), rows,
                                                                       max_unit_index)  # 添加订单时间节点，可以知道对应的订单
                                                self.work_status[start_zone][max_unit_index] = True  # 占据空闲生产单元
                                                timeline_one[rows] = time_line.current_time
                                                timeline_two[rows] = order_time + time_line.current_time
                                                timeline_history.append(timeline_one[:])
                                                timeline_history.append((start_zone,max_unit_index))
                                                timeline_history.append(timeline_two[:])
                                                timeline_history_3.append(timeline_one[:])

            agv_timeline_history.append(agv_timeline_one[:])
            agv_timeline_history.append(agv_order[:])
            agv_timeline_history.append(agv_timeline_two[:])
            # if point_type == 'start':
            #     timeline_history_3.append([-3] * num_rows)
            #     timeline_history_3.append([0] * num_rows)
            #     timeline_history_3.append(time_line.timeline[:])

            point_type, idx = time_line.get_next_point()
            # 我需要在这里求两个时间轴里面的最小时间节点（特殊情况：分别有一个时间节点在两个列表相同）
        order_time = max(use_time)
        return order_time, use_power

    def _initialize_function(self, idx):
        for zone, unit_count in zip(self.work_name, self.unit_states[idx]):
            self.machines_count[zone] = [0] * unit_count  # 为每个生产单元初始化机器数为 0
            self.work_status[zone] = [False] * unit_count  # 初始时，生产区是空闲的
            self.start_time[zone] = [0] * unit_count  # 初始时，没有开始时间
            self.end_time[zone] = [0] * unit_count  # 初始时，没有结束时间

        for zone, agv_count in zip(self.work_name, self.agv_count[idx]):
            self.agv_states[zone] = [False] * agv_count  # 初始时，生产区是空闲的
            self.agv_start_time[zone] = [0] * agv_count  # 初始时，没有开始时间
            self.agv_end_time[zone] = [0] * agv_count  # 初始时，没有结束时间

    def padding(self, sequence):
        """序列反填充"""
        sequence_idx = 0  # 追踪当前序列中的索引
        # 使用序列填充机器臂数量
        for zone in self.machines_count:
            for i in range(len(self.machines_count[zone])):
                if sequence_idx < len(list(str(sequence))):
                    # 将 sequence 中的数字逐个赋值到生产区的机器臂数量
                    # 把每个元素转换为字符串，然后逐个字符赋值
                    machines = list(str(sequence))  # 将数字转为字符串
                    self.machines_count[zone][i] = int(machines[sequence_idx])  # 给当前生产单元赋值机器数量
                    sequence_idx += 1

    def object_function_1(self, sequence, idx):  # 由序列改变字典，用于使用交叉变异修改机器臂分配后计算时间
        """初始化"""
        self._initialize_function(idx)
        self.padding(sequence)

        """计算每个订单的消耗，功率应该累加，但是时间不应该"""
        total_time_order, total_power_order = self.order_time_and_power(self.orders_list[idx], idx, '')

        return total_power_order, total_time_order

    def object_function_compare(self, sequence, idx, compare):  # 由序列改变字典，用于使用交叉变异修改机器臂分配后计算时间
        """初始化"""
        self._initialize_function(idx)
        self.padding(sequence)
        time_record = []
        energy_record = []
        weight_record = []
        counter_record = []
        # 模拟退火温度
        T = copy.deepcopy(self.SA_temperature[idx])
        # 降温速度
        a = 0.98
        # operators = [随机破坏修复， 后悔修复， 贪心修复]
        if compare == 4 or compare == 5:
            # 轮盘赌的初始权重
            operator_weight = [1, 1]
            if len(self.counter) - 1 < idx:
                # 算子的使用次数
                operator_counter = [0, 0]
            else:
                operator_counter = self.counter[idx]
            # 算子的初始得分
            operator_score = [1, 1]

        else:
            # 轮盘赌的初始权重
            operator_weight = [1, 1, 1]
            if len(self.counter) - 1 < idx:
                # 算子的使用次数
                operator_counter = [0, 0, 0]
            else:
                operator_counter = self.counter[idx][-1]

            # 算子的初始得分
            operator_score = [1, 1, 1]
        # 算子的挥发系数
        lambda_rate = 0.5
        # 开始ALNS迭代
        current_iteration = 0
        # 生成初始解
        first_order = copy.deepcopy(self.orders_list[idx])
        total_time_order, total_power_order = self.order_time_and_power(first_order, idx, 'picture')
        # 记录绘制甘特图的信息
        timeline_record = copy.deepcopy(self.timeline_record)
        # 同时清除原self.timeline_record的值
        self.timeline_record = {}
        # 将初始解设置为历史最优解和当前解
        best_value = (total_time_order, total_power_order, first_order, timeline_record)
        worst_value = (total_time_order, total_power_order)
        current_value = (total_time_order, total_power_order, first_order, timeline_record)
        time_record.append(best_value[0])
        energy_record.append(best_value[1])
        total_weight = sum(operator_weight)
        # 计算每个权重的占比
        weight_percentage = [(weight / total_weight) * 100 for weight in operator_weight]
        # 先试试次数
        weight_record.append(weight_percentage[:])
        counter_record.append(operator_counter[:])
        while current_iteration < iteration:
            if current_iteration % 10 == 0:
                print(current_iteration)
            self._initialize_function(idx)
            self.padding(sequence)
            # 使用当前订单作为原始订单
            order = copy.deepcopy(current_value[2])
            new_order, operator_idx = self.apply_ALNS(idx, order, operator_weight, operator_counter, compare)
            """计算每个订单的消耗，功率应该累加，但是时间不应该"""
            total_time_order, total_power_order = self.order_time_and_power(new_order, idx, 'picture')
            # 记录绘制甘特图的信息
            timeline_record = copy.deepcopy(self.timeline_record)
            # 同时清除原self.timeline_record的值
            self.timeline_record = {}
            if total_time_order >= worst_value[0] and total_time_order >= worst_value[1]:
                # 如果解比最差解差，修改最差解
                worst_value = (total_time_order, total_power_order)
            if total_time_order <= current_value[0] and total_power_order <= current_value[1]:
                # 如果新的解好于原来的解，就更新解
                current_value = (total_time_order, total_power_order, new_order, timeline_record)
                if total_time_order <= best_value[0] and total_power_order <= best_value[1]:
                    # 测试解为历史最优解，更新历史最优解，并设置最高的算子得分
                    best_value = (total_time_order, total_power_order, new_order, timeline_record)
                    operator_score[operator_idx] = 2.0
                elif total_time_order >=best_value[1] and total_power_order >=best_value[0]:
                    # 测试解不是历史最优解，但优于当前解，设置第三高的算子得分
                    operator_score[operator_idx] = 1.5
                else:
                    # 测试解与历史最优解相互支配，设置第二高的算子得分
                    best_value = (total_time_order, total_power_order, new_order, timeline_record)
                    operator_score[operator_idx] = 1.8

            elif (total_time_order >= current_value[0] and total_power_order >= current_value[1]):
                # 测试解被当前解支配，概率接受测试解
                # 计算delta
                # 归一化改进量（避免除零）
                time_range = max(worst_value[0] - best_value[0], 1e-6)
                energy_range = max(worst_value[1] - best_value[1], 1e-6)
                delta = - ((total_time_order - current_value[0]) / time_range + (total_power_order - current_value[1]) / energy_range) / 2
                prob = math.exp(delta / T)
                if random.random() < prob:
                    # 当前解优于测试解，但满足模拟退火逻辑，依然更新当前解，设置较低的算子得分
                    current_value = (total_time_order, total_power_order, new_order, timeline_record)
                    operator_score[operator_idx] = 0.8
                else:
                    # 当前解优于测试解，也不满足模拟退火逻辑，不更新当前解，设置最低的算子得分
                    operator_score[operator_idx] = 0.5
            else:
                # 测试解和当前解互相支配,接受当前解
                current_value = (total_time_order, total_power_order, new_order, timeline_record)
                if total_time_order > best_value[0] and total_power_order > best_value[1]:
                    # 如果测试解不如最优解
                    operator_score[operator_idx] = 1.2
                else:
                    # 测试解与最优解互相支配，接受最优解(与当前解相互支配不可能好于最优解)
                    best_value = (total_time_order, total_power_order, new_order, timeline_record)
                    operator_score[operator_idx] = 1.8

            if operator_idx == 0:
                # lambda_rate =
                operator_score[operator_idx] **= 1  # 通过指数放大差异
            else:
                # lambda_rate = 0.8
                # 更新算子的权重
                operator_score[operator_idx] **= 3  # 通过指数放大差异
            operator_weight[operator_idx] = round(
                operator_weight[operator_idx] * lambda_rate +(1 - lambda_rate) * operator_score[operator_idx] / operator_counter[operator_idx], 2)
            # 计算总和
            total_weight = sum(operator_weight)
            # 计算每个权重的占比
            weight_percentage = [round((weight / total_weight) * 100,2) for weight in operator_weight]

            # 记录更新过程
            time_record.append(best_value[0])
            energy_record.append(best_value[1])
            weight_record.append(weight_percentage[:])
            counter_record.append(operator_counter[:])
            # 降低温度
            T *= a
            current_iteration += 1
        total_time_order, total_power_order, best_order, timeline_record = best_value
        self.timeline_history.append(timeline_record['timeline_history'])
        self.timeline_history_1.append(timeline_record['timeline_history_1'])
        self.timeline_history_2.append(timeline_record['timeline_history_2'])
        self.timeline_history_3.append(timeline_record['timeline_history_3'])
        self.agv_timeline_history.append(timeline_record['agv_timeline_history'])
        value_length = len(self.iteration_value)
        if value_length - 1 < idx:
            self.iteration_value.append((time_record, energy_record))
            self.weight.append(weight_record)
            self.counter.append(counter_record)
        else:
            self.iteration_value[idx][0].extend(time_record)
            self.iteration_value[idx][1].extend(energy_record)
            self.weight[idx].extend(weight_record)
            self.counter[idx].extend(counter_record)

        self.SA_temperature[idx] = T
        return total_power_order, total_time_order, best_order

    """现在所计算出来的都是一个订单的时间和功率并且是用了最多机器臂情况下的结果"""

    def calculate_similarity(self, orders, idx):
        """进行订单顺序的ALNS时，相似性指标计算"""
        """我想存在矩阵里面"""
        similarity_matrix = np.zeros((len(orders), len(orders)))

        for i in range(len(orders)):
            for j in range(i + 1, len(orders)):  # 只比较后续的订单，避免重复比较
                obj_order = orders[i]  # 查找这个列表中的元素是否在其他列表中
                search_order = orders[j]  # 被查找的其他元素
                similarity = 0  # 临时相似值
                for index, element in enumerate(obj_order):
                    if element in search_order:  # 检查该元素是否在 search_order 中
                        position_unit = work_name.index(element)  # 找到该生产区的生产单元的数量
                        num_unit = self.unit_states[idx][position_unit]
                        position = search_order.index(element)  # 获取该元素在 search_order 中的位置
                        similarity += math.exp(- abs(position - index)) / num_unit  # e^|poi - poj|/number_生产单元
                similarity = similarity / (j - i)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        """将矩阵的每一行加起来就是某个订单的总体相似值"""
        similarity_list = similarity_matrix.sum(axis=1)

        return similarity_list

    def insert_and_evaluate(self, order_sequence, new_order, idx):
        """插入并计算每一个排列组合插入到订单序列中的适配性，来选择如何插入"""
        """new_order为单括号列表[],order_sequence为双括号列表[ [] ]"""
        # 创建一个得分数组，初始化所有值为0
        scores_array = [0 for _ in range(len(order_sequence) + 1)]
        # 遍历每个可能的插入点
        for i in range(len(order_sequence) + 1):
            # 在第i个订单后插入新订单
            new_sequence = order_sequence[:i] + [new_order] + order_sequence[i:]
            # 计算这个新序列的指标
            metric = self.calculate_metric(new_sequence, i, idx)
            # 存储指标
            scores_array[i] = metric

        return scores_array  # 返回分数

    def calculate_metric(self, sequences, index, idx):
        """负责计算排列组合后的单个订单后悔算子的各个位置的指标"""
        """通过调用之前的相似性计算，直接选择需要求的订单索引处的值，即为所求"""
        metric_list = self.calculate_similarity(sequences, idx)
        metric = metric_list[index]
        return metric

    def find_key_and_list_for_index(self, data_dict, index):
        current_index = 0
        for key, lists in data_dict.items():
            for lst in lists:
                if current_index == index:
                    return key, lst  # 返回找到的键和列表
                current_index += 1
        return None, None  # 如果没有找到，返回None

    def apply_ALNS(self, idx, orders, operator_weight, operator_counter,compare):
        """
        使用ALNS算法优化订单顺序
        """
        now_order = copy.deepcopy(orders)
        if compare == 4:
            operators = [self.apply_random, self.greedy_repair]
        elif compare == 5:
            operators = [self.apply_random, self.apply_regret_repair]
        else:
            operators = [self.apply_random, self.apply_regret_repair, self.greedy_repair]
        # 使用轮盘赌选择算子的索引
        selected_operator, operator_idx = roulette_wheel_selection(operators, operator_weight)
        # 根据选择的算子进行操作
        new_order = selected_operator(idx, now_order)
        operator_counter[operator_idx] += 1
        return new_order, operator_idx

    def apply_random(self, idx, now_order):
        new_order = copy.deepcopy(now_order)
        random.shuffle(new_order)
        return new_order

    def apply_regret_repair(self, idx, now_order):
        # best_order = copy.deepcopy(self.orders_list[idx])  # 初始订单
        regret_matching_operator = []  # 要使用后悔修复算子的元素组成的列表
        similarity = self.calculate_similarity(now_order, idx)  # 存储订单相似值的列表
        num_destruction = int(len(now_order) * similarity_percent)  # 需要破坏的订单个数
        # 找到最大的部分元素
        max_values = heapq.nlargest(num_destruction, similarity)

        # 找到这些最大值的索引
        indices = [i for i, value in enumerate(similarity) if value in max_values]
        for index in indices:
            regret_matching_operator.append(now_order[index])  # 将最大值对应的订单添加到新列表

        """这里准备进行后悔修复，先在原先订单中删除待破坏的订单"""
        new_order = copy.deepcopy(now_order)  # 列表存储着订单，之后进行删除指标较高的订单
        for index in sorted(indices, reverse=True):  # reverse=True 确保从后往前删除
            del new_order[index]

        """接下来我要来破坏订单内部的工艺流程"""
        """决定将工艺流程暂定为[0,1,xxxx,6],所以所有的情况是xxxx的内部组合"""
        while regret_matching_operator:  # 如果待插入列表为空，即完成插入
            # 定义字典来存储所有排列组合的情况
            permutations_dict = {}
            for idx, lst in enumerate(regret_matching_operator):
                # 从索引2到倒数第二个元素（不包括最后的6）
                disruption = lst[2:-1]
                # 生成xxxx的所有排列
                permutations = list(itertools.permutations(disruption))
                # total_permutations += len(permutations)
                permutations = [[regret_matching_operator[idx][0]] + [regret_matching_operator[idx][1]]
                                + list(perm) + [regret_matching_operator[idx][-1]] for perm in permutations]
                # 存储排列,现在所有的排列情况都存在了字典中
                permutations_dict[idx] = permutations

            # 遍历字典，并对每个排列列表进行操作
            total_rh_value = []  # 用来存储每个排列的后悔值，方便后续比较
            row_vectors = []  # 存储每个key的permutations对应的行向量
            for key, permutations in permutations_dict.items():
                """对每一个订单的排列组合进行判断"""
                rh_value = 0  # 计算后悔值
                for perm in permutations:
                    # 得到了相似性的值的数组
                    array = self.insert_and_evaluate(new_order, perm, idx)
                    # 使用 sorted() 函数对列表进行排序
                    sorted_array = sorted(array)
                    # 后悔修复算子优解个数,为待插入个数
                    rh_number = len(regret_matching_operator)

                    """如果为1，后悔值就会为0"""
                    if rh_number == 1:
                        rh_number += 1
                    # 获取最小的 k 个数
                    smallest_array = sorted_array[:rh_number]
                    # 求k步的后悔值
                    for small_array in smallest_array:
                        rh_value += (small_array - smallest_array[0])
                    total_rh_value.append(rh_value)
                    # 记录所有的值以及索引
                    row_vectors.append(array)

            max_rh_value = max(total_rh_value)  # 找到每段排列组合之后的插入后悔值的最大值
            max_index = total_rh_value.index(max_rh_value)

            cope_array = row_vectors[max_index]  # 需要处理的适应值列表
            min_adapt_value = min(cope_array)  # 该列表适应值最小的值
            next_insert_position = cope_array.index(min_adapt_value)  # 订单待插入的位置

            # 找到了下一个应该插入的订单的样式,需要插入的列表在字典中的位置
            found_key, found_lst = self.find_key_and_list_for_index(permutations_dict, max_index)
            next_insert_order = found_lst  # 待插入订单
            """完成插入，并修改regret_matching_operator"""
            new_order.insert(next_insert_position, next_insert_order)
            del regret_matching_operator[found_key]

        return new_order

    def greedy_repair(self, idx, now_order):
        """
        使用ALNS算法优化订单顺序
        """
        """初始化最佳时间和能耗"""
        # best_order = copy.deepcopy(self.orders_list[idx])  # 初始订单
        regret_matching_operator = []  # 要使用后悔修复算子的元素组成的列表
        similarity = self.calculate_similarity(now_order, idx)  # 存储订单相似值的列表
        num_destruction = int(len(now_order) * similarity_percent)  # 需要破坏的订单个数
        # 找到最大的部分元素
        max_values = heapq.nlargest(num_destruction, similarity)

        # 找到这些最大值的索引
        indices = [i for i, value in enumerate(similarity) if value in max_values]
        for index in indices:
            regret_matching_operator.append(now_order[index])  # 将最大值对应的订单添加到新列表

        """这里准备进行后悔修复，先在原先订单中删除待破坏的订单"""
        new_order = copy.deepcopy(now_order)  # 列表存储着订单，之后进行删除指标较高的订单
        for index in sorted(indices, reverse=True):  # reverse=True 确保从后往前删除
            del new_order[index]

        """接下来我要来破坏订单内部的工艺流程"""
        """决定将工艺流程暂定为[0,1,xxxx,6],所以所有的情况是xxxx的内部组合"""
        while regret_matching_operator:  # 如果待插入列表为空，即完成插入
            # 定义字典来存储所有排列组合的情况
            permutations_dict = {}
            # total_permutations = 0  # 存储所有排列的数量
            for idx, lst in enumerate(regret_matching_operator):
                # 从索引2到倒数第二个元素（不包括最后的6）
                disruption = lst[2:-1]
                # 生成xxxx的所有排列
                permutations = list(itertools.permutations(disruption))
                # total_permutations += len(permutations)
                permutations = [[regret_matching_operator[idx][0]] + [regret_matching_operator[idx][1]]
                                + list(perm) + [regret_matching_operator[idx][-1]] for perm in permutations]
                # 存储排列,现在所有的排列情况都存在了字典中
                permutations_dict[idx] = permutations
            # 遍历字典，并对每个排列列表进行操作
            total_min_value = []  # 用来存储每个排列的最小适应值，方便后续比较
            total_min_value_index = []  # 用来存储每个排列的最小适应值的索引，方便后续比较
            for key, permutations in permutations_dict.items():
                """对每一个订单的排列组合进行判断"""
                row_vectors = []  # 存储每个key的permutations对应的行向量
                for perm in permutations:
                    # 得到了相似性的值的数组
                    array = self.insert_and_evaluate(new_order, perm, idx)
                    # 添加到列表中
                    row_vectors.append(array)
                matrix = np.vstack(row_vectors)
                """接下来是对矩阵进行最小值及其索引的寻找与储存"""
                min_value = np.min(matrix)  # 找到矩阵中的最小值
                total_min_value.append(min_value)  # 存入列表，方便之后比较
                min_positions = np.where(matrix == min_value)  # 寻找最小值位置
                total_min_value_index.append(list(zip(min_positions[0], min_positions[1])))

                """这样比较每个最小值得到插入点，然后迭代就可以完成"""
            min_value = min(total_min_value)  # 找到每段排列组合之后的插入最小值的最小值
            min_index = total_min_value.index(min_value)
            next_insert_position = total_min_value_index[min_index]  # 存储了所有适应值中的最小值在其矩阵中所在位置
            # 找到了下一个应该插入的订单的样式
            next_insert_order = permutations_dict[min_index][next_insert_position[0][0]]
            """完成插入，并修改regret_matching_operator"""
            new_order.insert(next_insert_position[0][1], next_insert_order)
            del regret_matching_operator[min_index]

        return new_order

def roulette_wheel_selection(operators, operator_weight):
    # 计算权重总和
    total_weight = sum(operator_weight)

    # 计算每个个体的选择概率
    probabilities = [weight / total_weight for weight in operator_weight]

    # 计算累积概率
    cumulative_probabilities = []
    cumulative_sum = 0
    for prob in probabilities:
        cumulative_sum += prob
        cumulative_probabilities.append(cumulative_sum)

    # 随机选择一个值
    r = random.random()  # 生成一个[0, 1)之间的随机数

    # 根据累积概率选择算子
    for i, cum_prob in enumerate(cumulative_probabilities):
        if r <= cum_prob:
            return operators[i], i
