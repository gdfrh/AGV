from Config import *
from order import *
import random
import time
from Util import *


class Arm:
    def __init__(self, work_name, unit_numbers, total_machines, machine_power,num_orders):
        # 初始化生产区名称和单元数
        self.work_name = work_name
        self.unit_numbers = unit_numbers
        self.total_machines = total_machines  # 总机器数量
        self.machine_power = machine_power  # 每台机器的功率
        self.machines_count = {}  # 创建字典来存储每个生产单元的机器数
        self.order_manager = OrderManager(work_name, num_orders)
        self.orders = self.order_manager.get_orders()  # 调用 OrderManager来显示订单
        # print(self.orders)
        """用False表示空闲，True表示忙碌"""
        self.work_status = {}  # 用来判断生产区的生产单元是否在工作
        self.start_time = {}  # 用来记录生产区工作的开始时间
        self.end_time = {}  # 用来记录生产区工作的结束时间
        self._initialize_cells()  # 初始化生产区的机器数

        self.state_change()  # 不断判断生产单元的状态并改变

    def _initialize_cells(self):
        """初始化生产区及其对应的机器数"""
        for zone, unit_count in zip(self.work_name, self.unit_numbers):
            self.machines_count[zone] = [0] * unit_count  # 为每个生产单元初始化机器数为 0
            self.work_status[zone] = [False] * unit_count  # 初始时，生产区是空闲的
            self.start_time[zone] = [None] * unit_count  #初始时，没有开始时间
            self.end_time[zone] = [None] * unit_count  #初始时，没有结束时间
            # print(self.work_status[zone])


    def distribute_machines_randomly(self):
        """每次初始化解的时候需要对work_state也要初始化"""
        self._initialize_cells()
        """随机将机器分配到生产单元，并显示每个生产区分配的机器数量"""
        available_units = []  # 存储所有生产单元的可用位置（生产区 + 单元索引）

        # 1. 首先为每个生产单元分配一个机器
        for zone in self.machines_count:
            for i in range(len(self.machines_count[zone])):
                self.machines_count[zone][i] = 1  # 每个单元至少分配1台机器
                available_units.append((zone, i))  # 存储可用单元

        # 剩余机器数量
        remaining_machines = self.total_machines - sum([sum(self.machines_count[zone]) for zone in self.machines_count])

        # 2. 随机分配剩余的机器
        while remaining_machines > 0:
            zone, unit_index = random.choice(available_units)  # 随机选择一个单元
            self.machines_count[zone][unit_index] += 1  # 为该单元分配一台机器
            remaining_machines -= 1  # 剩余机器数减少1

    def display_machine_count(self):
        """返回每个生产区中各单元的机器数，并存储到列表中"""
        machine_count_list = []   # 列表
        for zone, units in self.machines_count.items():
            # 将每个生产区的机器数存储到字典中
            machine_count_list += units  # 每个单元的机器数
            # print(
            #     f"{zone}: {', '.join([str(unit) for unit in units])} 台机器")
        merged_str = ''.join(map(str, machine_count_list))
        # 将连接后的字符串转换为整数
        merged_number = int(merged_str)
        machine_count_list_renew = merged_number
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

        for i in range(len(order)):
            start_zone = order[i]
            if False in self.work_status[start_zone]:
                """如果该生产区存在空闲单元"""
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
                if i < len(order) - 1:
                    end_zone = order[i + 1]
                    transport_time = self.calculate_transport_time(start_zone, end_zone, distance_matrix, work_name_order, vga_speed)
                    order_total_time += transport_time  # 加上运输时间
                # 3. 只有等待小车的时间

            else:#生产单元全部忙碌
                """需要获得等待时间再加上工作时间"""
                print()

        return order_total_time, order_total_power

    def state_change(self):
        """不断地查看状态是否变化"""
        while True:
            """对于每个生产单元，如果当前时间为任务结束时间，就改变生产单元状态为False空闲"""
            for zone in self.machines_count:
                for i in range(len(self.machines_count[zone])):
                    if time.time() == self.end_time[zone][i]:
                        self.work_status[zone][i] = False


    def object_function(self, sequence):  # 由序列改变字典，用于使用交叉变异修改机器臂分配后计算时间
        """初始化"""
        self._initialize_cells()

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

        """任务的指定,需要修改"""
        # tasks = {}
        #
        # for zone, unit_count in zip(self.work_name,self.unit_numbers):
        #     tasks[zone] = []
        #     for _ in range(unit_count):
        #         # 为每个单元生成随机任务列表
        #         tasks[zone].append([self.generate_task() for _ in range(3)])  # 每个生产单元生成3个任务

        """能量，时间计算"""
        total_time, total_power = 0, 0
        for order in self.orders:
            """计算每个订单的消耗"""
            total_time_order, total_power_order = self.order_time_and_power(order)
            total_time += total_time_order
            total_power += total_power_order
        return total_time, total_power
        # total_energy, total_time = 0, 0
        # for zone in self.work_name:
        #     for unit_index in range(len(self.machines_count[zone])):
        #         # 获取该生产单元的机器数量
        #         machine_count = self.machines_count[zone][unit_index]
        #
        #         for task in tasks[zone][unit_index]:
        #             run_time = task['run_time']
        #             run_power = task['run_power']
        #             sleep_time = task['sleep_time']
        #             sleep_power = task['sleep_power']
        #             run_time_renew, run_power_renew = self.calculate_reduction(run_time, run_power, zone, machine_count)
        #
        #             # 计算该任务的能量消耗
        #             task_energy = self.calculate_task_energy(run_time_renew, run_power_renew, sleep_time, sleep_power) * machine_count
        #             total_energy += task_energy
        #             total_time += run_time_renew + sleep_time
        #
        # return total_energy, total_time
"""现在所计算出来的都是一个订单的时间和功率并且是用了最多机器臂情况下的结果，并且时间是累加计算的，应该得计算最后结束就行"""

