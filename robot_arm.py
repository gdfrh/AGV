from Config import *
from order import *
import random
import threading
import time
from Util import *



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
        print(self.orders)
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

        # # 3. 对每个生产区的生产单元按机器数量从大到小排序
        # for zone in self.machines_count:
        #     # 排序每个生产区的单元，按机器数量从大到小
        #     self.machines_count[zone] = sorted(self.machines_count[zone], reverse=True)

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

        reduction_factor_time = 0  #初始化
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

    def find_true_min_time(self, zone):
        """找出某个生产区中状态为 True 的生产单元中，所需等待时间最短的"""
        min_wait_time = float('inf')
        min_unit_index = -1  # 记录等待时间最短的生产单元的索引

        for i, status in enumerate(self.work_status[zone]):
            if status:  # 如果当前生产单元处于忙碌状态
                if self.end_time[zone][i] - time.time() < min_wait_time:
                    min_wait_time = self.end_time[zone][i] - time.time()
                    min_unit_index = i

        # 返回等待时间最短的生产单元索引及其等待时间
        if min_unit_index != -1:
            return min_unit_index, min_wait_time
        else:
            return None  # 如果没有空闲单元

    def find_true_min_time_agv(self, zone):
        """找出某个生产区中状态为 True 所需等待时间最短的小车"""
        min_wait_time = float('inf')
        min_unit_index = -1  # 记录等待时间最短的生产单元的索引

        for i, status in enumerate(self.agv_states[zone]):
            if status:  # 如果当前小车处于忙碌状态
                if self.agv_end_time[zone][i] - time.time() < min_wait_time:
                    min_wait_time = self.agv_end_time[zone][i] - time.time()
                    min_unit_index = i

        # 返回等待时间最短的生产单元索引及其等待时间
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
        # 遍历订单的每个生产区，计算总时间
        # for i in order:
        #     units = self.machines_count[i]
        #
        #     """这里我们先找到最多机器臂的生产单元和机器臂数量，再根据数量对时间和功率的影响修改时间和功率"""
        #     max_machines.append(max(units))  # 找到最大的机器臂数量
        #     #max_index = units.index(max_machines)  # 找到最大值的索引
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
                min_unit_index, min_wait_time = self.find_true_min_time(start_zone)
                self.work_status[start_zone][min_unit_index] = False

            if False in self.work_status[start_zone]:
                """如果该生产区存在空闲单元,暂时寻找最大机器臂数量的生产单元"""
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
                    min_unit_index_agv, min_wait_time_agv = self.find_true_min_time_agv(start_zone)
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

    def object_function(self, sequence, idx):  # 由序列改变字典，用于使用交叉变异修改机器臂分配后计算时间
        """初始化"""
        for zone, unit_count in zip(self.work_name, self.unit_states[idx]):
            self.machines_count[zone] = [0] * unit_count  # 为每个生产单元初始化机器数为 0
            self.work_status[zone] = [False] * unit_count  # 初始时，生产区是空闲的
            self.start_time[zone] = [0] * unit_count  # 初始时，没有开始时间
            self.end_time[zone] = [0] * unit_count  # 初始时，没有结束时间

        for zone, agv_count in zip(self.work_name, self.agv_count[idx]):
            self.agv_states[zone] = [False] * agv_count  # 初始时，生产区是空闲的
            self.agv_start_time[zone] = [0] * agv_count  # 初始时，没有开始时间
            self.agv_end_time[zone] = [0] * agv_count  # 初始时，没有结束时间

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

        """ALNS计算能量，时间"""
        best_order, best_time, best_power = self.apply_ALNS(iterations=30)

        return best_power, best_time

    """现在所计算出来的都是一个订单的时间和功率并且是用了最多机器臂情况下的结果"""

    """这里要使用破坏修复算子"""
    def random_destruction_fix(self, orders):
        """
        随机交换两个订单的位置，随机破坏，随机修复
        """
        new_orders = orders[:]
        i, j = random.sample(range(len(orders)), 2)  # 随机选两个索引
        new_orders[i], new_orders[j] = new_orders[j], new_orders[i]  # 交换
        return new_orders

    def reverse(self, orders):
        """
        随机反转一段订单顺序
        """
        new_orders = orders[:]
        i, j = random.sample(range(len(orders)), 2)  # 随机选两个索引
        if i > j:
            i, j = j, i  # 保证 i < j
        new_orders[i:j+1] = reversed(new_orders[i:j+1])  # 反转区间
        return new_orders

    def greedy_destruction(self, list1, list2):
        """破坏时间最长的"""
        # 计算需要获取的前 % 的元素数量
        top_percent_count = int(len(list1) * greedy_percent)

        # 获取总时间列表中每个元素的索引和对应的值
        indexed_list = list(enumerate(list1))

        # 按照总时间的值降序排序
        indexed_list.sort(key=lambda x: x[1], reverse=True)

        # 取前 % 最大值的索引
        top_percent_indices = [index for index, _ in indexed_list[:top_percent_count]]

        # 存储删除的元素
        removed_elements = []

        # 按照从大的索引到小的索引删除元素
        for index in sorted(top_percent_indices, reverse=True):
            removed_elements.append(list2[index])  # 存储删除的元素
            del list2[index]  # 删除元素

        # 随机顺序将删除的元素插入到列表尾部
        random.shuffle(removed_elements)  # 随机打乱删除的元素

        # 将删除的元素插入到列表尾部
        list2.extend(removed_elements)

        return list2
    def select_neighbor(self, orders):
        """
        随机选择一个邻域操作（比如交换或反转）来改变订单顺序
        """
        # 随机选择操作：50%概率进行交换，50%概率进行反转
        if random.random() < 0.5:
            return self.random_destruction_fix(orders)
        else:
            return self.reverse(orders)

    def apply_ALNS(self, iterations=100):
        """
        使用ALNS算法优化订单顺序
        """
        """初始化最佳时间和能耗"""
        best_time, best_power = float('inf'), float('inf')
        best_order = self.orders  # 初始订单
        for _ in range(iterations):
            # 随机选择一个邻域操作
            neighbor_order = self.select_neighbor(best_order)

            """能量，时间计算"""
            total_time, total_power = 0, 0
            total_time_list = []
            # 计算邻域解的总时间和总功率
            """计算每个订单的消耗，功率应该累加，但是时间不应该"""
            for order in neighbor_order:
                total_time_order, total_power_order = self.order_time_and_power(order)
                total_time_list.append(total_time_order)
                total_power += total_power_order

            total_time = max(total_time_list)
            # 如果邻域解的时间或功率更好，则更新最优解
            if total_time < best_time and total_power < best_power:
                best_order = neighbor_order
                best_time = total_time
                best_power = total_power

                # 更新当前解
                best_order = neighbor_order
            if random.random() < 0.5:
                # 概率使用贪婪破坏修复
                best_order = self.greedy_destruction(total_time_list, neighbor_order)

        return best_order, best_time, best_power