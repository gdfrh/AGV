from Config import *
from order import *
import random
import time
import numpy as np
import math
import heapq
import itertools
import copy


class Arm:
    def __init__(self, work_name, unit_numbers, total_machines, machine_power,num_orders):
        # 初始化生产区名称和单元数
        self.work_name = work_name
        self.unit_numbers = unit_numbers  # 初始的生产单元个数
        self.unit_states = []  # 后续每个序列的生产单元个数
        self.total_machines = total_machines  # 总机器数量
        self.machine_power = machine_power  # 每台机器的功率
        self.machines_count = {}  # 创建字典来存储每个生产单元的机器数
        self.order_manager = OrderManager(work_name, num_orders)
        self.orders = self.order_manager.get_orders()  # 调用 OrderManager来显示订单
        """用False表示空闲，True表示忙碌"""
        self.work_status = {}  # 用来判断生产区的生产单元是否在工作
        self.start_time = {}  # 用来记录生产区工作的开始时间
        self.end_time = {}  # 用来记录生产区工作的结束时间
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
        agv_distribute = []
        for zone, non_zero_count in non_zero_machines_count.items():
            # 计算该生产区应该分配的小车数量
            proportion = non_zero_count / total_non_zero
            car_count_for_zone = round(proportion * total_agv)  # 根据比例计算，并取整
            agv_distribute.append(car_count_for_zone)
        self.agv_count.append(agv_distribute)

        # 3. 对每个生产区的生产单元按机器数量从大到小排序
        for zone in self.machines_count:
            # 排序每个生产区的单元，按机器数量从大到小
            self.machines_count[zone] = sorted(self.machines_count[zone], reverse=True)

    def display_machine_count(self):
        """返回每个生产区中各单元的机器数，并存储到列表中"""
        machine_count_list = []   # 列表
        for zone, units in self.machines_count.items():
            # 将每个生产区的机器数存储到字典中
            machine_count_list += units  # 每个单元的机器数
        # print(machine_count_list)
        merged_str = ''.join(map(str, machine_count_list))
        # 将连接后的字符串转换为整数
        merged_number = int(merged_str)
        """前导0的问题，不足位数补0"""
        # 先计算总单元数
        total_units = sum(self.unit_numbers)
        machine_count_list_renew = f"{merged_number:0{total_units}d}"
        # machine_count_list_renew = merged_number
        return machine_count_list_renew

    def calculate_reduction(self, initial_time, initial_power, zone_name, machine_count):
        """
        根据机器臂数量来改变消耗
        参数:
        initial_time (float): 初始运行时间
        initial_power (float):初始运行功率
        zone (str): 生产区名称
        machine_count (int): 当前该生产单元的机器数量

        返回:
        float: 调整后的运行时间
        """

        reduction_factor_time = 0  # 初始化
        reduction_factor_power = 0
        temp_idx = 0
        for zone in self.work_name:

            if zone == zone_name:
                reduction_factor_time = reduction_factor_time_list[temp_idx]
                reduction_factor_power = reduction_factor_power_list[temp_idx]
                break
            temp_idx += 1

        return initial_time * ((1 - reduction_factor_time )**machine_count)\
              ,initial_power*((1 - reduction_factor_power )**machine_count)

    def calculate_task_energy(self, run_time, run_power, sleep_time, sleep_power):
        """
        计算单个任务的能量消耗

        参数:
        run_time (float): 执行工艺任务的运行时间 (秒)
        run_power (float): 执行工艺任务的运行功率 (瓦特)
        sleep_time (float): 休眠时间 (秒)
        sleep_power (float): 休眠功率 (瓦特)
        machine_count (int): 该生产单元的机器数量

        返回:
        float: 该任务的总能量消耗 (焦耳, J)
        """
        # 单个任务的能量消耗
        task_energy = (run_time * run_power) + (sleep_time * sleep_power)
        return task_energy

    def calculate_processing_time(self):
        """计算生产区的处理时间"""
        return processing_time

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
        :param obj_type: 'unit表示生产单元，'agv表示小车
        """
        min_wait_time = float('inf')
        min_unit_index = -1  # 记录等待时间最短的生产单元的索引

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
                remaining_time = end_time_list[i] - time.time()
                if remaining_time < min_wait_time:
                    min_wait_time = remaining_time
                    min_unit_index = i

        # 返回等待时间最短的单元索引及其等待时间
        if min_unit_index != -1:
            return min_unit_index, min_wait_time
        else:
            return None  # 如果没有空闲单元

    def find_random_idle_agv(self, start_zone):
        """随机找到一个空闲的小车，并返回其索引"""
        # 获取当前生产区的工作状态
        agv_work_status = self.agv_states[start_zone]

        # 找到所有空闲的小车的索引（状态为 False）
        idle_cars = [index for index, status in enumerate(agv_work_status) if not status]

        return idle_cars

    def order_time_and_power(self, order):
        """计算一个订单的处理时间和功率消耗，包括生产区时间和运输时间"""
        order_total_time = 0  # 一个订单完成所需时间
        order_total_power = 0  # 一个订单完成所需功率

        """遍历所有生产区和单元，将机器臂数量为 0 的生产单元的状态改为忙碌，避免出现选择0个机器臂的生产单元进行操作"""
        for zone in self.machines_count:
            for i in range(len(self.machines_count[zone])):
                if self.machines_count[zone][i] == 0:  # 如果该生产单元的机器臂数量为 0
                    self.work_status[zone][i] = True  # 将该生产单元状态设为忙碌
                    self.end_time[zone][i] = float('inf')  # 设置一个很大的结束时间，确保不会被选中

        for i in range(len(order)):
            start_zone = order[i]
            min_unit_index, min_wait_time = 0, 0
            min_unit_index_agv, min_wait_time_agv = 0, 0

            if False not in self.work_status[start_zone]:  # 生产单元全部忙碌
                """需要获得等待时间再加上工作时间"""
                min_unit_index, min_wait_time = self.find_min_wait_time(start_zone,'unit')
                self.work_status[start_zone][min_unit_index] = False

            if False in self.work_status[start_zone]:
                """如果该生产区存在空闲单元,暂时寻找最大机器臂数量的生产单元"""
                """"""
                """"""
                """这里要改成找到加权之后的最低的生产单元"""
                max_unit_index, max_machines = self.find_false_max_machines(start_zone)  # 返回空闲生产单元机器臂最大值索引和数量
                self.work_status[start_zone][max_unit_index] = True  # 占据了这个生产单元
                self.start_time[start_zone][max_unit_index] = time.time()  # 将开始时间记录下来
                # print(self.work_status)
                order_total_time_renew, order_total_power_renew = self.calculate_reduction(processing_time['run_time'], processing_time['run_power'], start_zone, max_machines)
                order_total_time += order_total_time_renew + processing_time['sleep_time']
                order_total_power += self.calculate_task_energy(order_total_time_renew, order_total_power_renew, processing_time['sleep_time'], processing_time['sleep_power']) * max_machines
                # 记录任务结束时间
                self.end_time[start_zone][max_unit_index] = self.start_time[start_zone][max_unit_index] + order_total_time_renew + processing_time['sleep_time']

                # 2. 计算运输时间
                if False not in self.agv_states[start_zone]:  # 该生产区的小车全部忙碌
                    """如果该生产区不存在空闲小车"""
                    min_unit_index_agv, min_wait_time_agv = self.find_min_wait_time(start_zone, 'agv')
                    self.agv_states[start_zone][min_unit_index] = False

                if False in self.work_status[start_zone]:
                    """如果该生产区存在空闲小车"""
                    agv_index = self.find_random_idle_agv(start_zone)  # 返回空闲生产单元机器臂最大值索引和数量
                    self.agv_states[start_zone][agv_index[0]] = True  # 占据了这个生产单元
                    self.agv_start_time[start_zone][agv_index[0]] = time.time()  # 将开始时间记录下来
                    transport_time = 0
                    if i < len(order) - 1:
                        end_zone = order[i + 1]
                        transport_time = self.calculate_transport_time(start_zone, end_zone, distance_matrix, work_name_order, agv_speed)
                        order_total_time += transport_time  # 加上运输时间
                        """离开车间"""
                    if i == len(order) - 1:
                        # Step 1: 创建一个反向字典，将生产区汉字名称映射到数字索引
                        zone_to_index = {name: index for index, name in work_name_order.items()}
                        # Step 2: 使用反向字典，将汉字生产区转换为数字索引
                        start_index = zone_to_index[start_zone]
                        # Step 3: 使用转换后的数字索引从距离矩阵中获取距离
                        distance = distance_matrix[start_index][-1]
                        # Step 4: 计算运输时间（假设速度是已知的，单位为距离/时间）
                        transport_time = distance / agv_speed  # 时间 = 距离 / 速度
                        order_total_time += transport_time  # 加上运输时间

                    self.agv_end_time[start_zone][agv_index[0]] = self.agv_start_time[start_zone][agv_index[0]] + transport_time * 2

                # 3. 工作区生产单元等待时间
                order_total_time += min_wait_time

                # 4. 工作区小车等待时间
                order_total_time += min_wait_time_agv

        return order_total_time, order_total_power

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

        """计算最终所需时间和能耗"""
        total_time, total_power = 0, 0
        total_time_list = []

        """计算每个订单的消耗，功率应该累加，但是时间不应该"""
        for order in self.orders:
            total_time_order, total_power_order = self.order_time_and_power(order)
            total_time_list.append(total_time_order)
            total_power += total_power_order

        best_time = max(total_time_list)
        best_power = total_power

        return best_power, best_time

    def object_function_2(self, sequence, idx):  # 由序列改变字典，用于使用交叉变异修改机器臂分配后计算时间
        """初始化"""
        self._initialize_function(idx)
        self.padding(sequence)

        """ALNS计算能量，时间"""
        best_order = self.apply_ALNS()
        """计算最终所需时间和能耗"""
        total_time, total_power = 0, 0
        total_time_list = []

        """计算每个订单的消耗，功率应该累加，但是时间不应该"""
        for order in best_order:
            total_time_order, total_power_order = self.order_time_and_power(order)
            total_time_list.append(total_time_order)
            total_power += total_power_order

        best_time = max(total_time_list)

        best_power = total_power

        return best_power, best_time, best_order

    """现在所计算出来的都是一个订单的时间和功率并且是用了最多机器臂情况下的结果"""

    def calculate_similarity(self, orders):
        """进行订单顺序的ALNS时，相似性指标计算"""
        """我想存在矩阵里面"""
        similarity_matrix = np.zeros((len(orders), len(orders)))

        for i in range(len(orders)):
            for j in range(i + 1, len(orders)):  # 只比较后续的订单，避免重复比较
                obj_order = orders[i]  # 查找这个列表中的元素是否在其他列表中
                search_order = orders[j]  # 被查找的其他元素
                # # 在这里进行比较操作
                # if obj_order == search_order:  # 如果订单完全相同，相似值设为最大
                #     """越近越相似"""
                #     similarity_matrix[i, j] = 10000 / (j - i)
                #     similarity_matrix[j, i] = 10000 / (j - i)
                # else:
                similarity = 0  # 临时相似值
                for index, element in enumerate(obj_order):
                    if element in search_order:  # 检查该元素是否在 search_order 中
                        position_unit = work_name.index(element)  # 找到该生产区的生产单元的数量
                        num_unit = self.unit_states[-1][position_unit]
                        position = search_order.index(element)  # 获取该元素在 search_order 中的位置
                        similarity += math.exp(- abs(position - index)) / num_unit  # e^|poi - poj|/number_生产单元
                similarity = similarity / (j - i)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        """将矩阵的每一行加起来就是某个订单的总体相似值"""
        similarity_list = similarity_matrix.sum(axis=1)

        return similarity_list

    def insert_and_evaluate(self, order_sequence, new_order):
        """插入并计算每一个排列组合插入到订单序列中的适配性，来选择如何插入"""
        """new_order为单括号列表[],order_sequence为双括号列表[ [] ]"""
        # 创建一个得分数组，初始化所有值为0
        scores_array = [0 for _ in range(len(order_sequence) + 1)]
        # 遍历每个可能的插入点
        for i in range(len(order_sequence) + 1):
            # 在第i个订单后插入新订单
            new_sequence = order_sequence[:i] + [new_order] + order_sequence[i:]
            # 计算这个新序列的指标
            metric = self.calculate_metric(new_sequence, i)
            # 存储指标
            scores_array[i] = metric

        return scores_array  # 返回分数

    def calculate_metric(self, sequences, index):
        """负责计算排列组合后的单个订单后悔算子的各个位置的指标"""
        """通过调用之前的相似性计算，直接选择需要求的订单索引处的值，即为所求"""
        metric_list = self.calculate_similarity(sequences)
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

    def apply_ALNS(self):
        """
        使用ALNS算法优化订单顺序
        """
        """初始化最佳时间和能耗"""
        best_order = copy.deepcopy(self.orders)  # 初始订单
        for _ in range(iterations):
            regret_matching_operator = []  # 要使用后悔修复算子的元素组成的列表
            similarity = self.calculate_similarity(best_order)  # 存储订单相似值的列表
            num_destruction = int(len(best_order) * similarity_percent)  # 需要破坏的订单个数
            # 找到最大的部分元素
            max_values = heapq.nlargest(num_destruction, similarity)

            # 找到这些最大值的索引
            indices = [i for i, value in enumerate(similarity) if value in max_values]
            for index in indices:
                regret_matching_operator.append(best_order[index])  # 将最大值对应的订单添加到新列表

            """这里准备进行后悔修复，先在原先订单中删除待破坏的订单"""
            new_order = best_order  # 列表存储着订单，之后进行删除指标较高的订单
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
                        array = self.insert_and_evaluate(new_order, perm)
                        # 使用 sorted() 函数对列表进行排序
                        sorted_array = sorted(array)
                        # 后悔修复算子次优解个数,为待插入个数
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

            """将插入好的列表返回为最好列表，进行迭代"""
            best_order = new_order

        return best_order

