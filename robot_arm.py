from Config import *
import random


class Arm:
    def __init__(self, work_name_up, work_name_down, unit_numbers_up, unit_numbers_down, total_machines, machine_power,complexity):
        # 初始化生产区名称和单元数
        self.work_name_up = work_name_up
        self.work_name_down = work_name_down
        self.unit_numbers_up = unit_numbers_up
        self.unit_numbers_down = unit_numbers_down
        self.total_machines = total_machines  # 总机器数量
        self.machine_power = machine_power  # 每台机器的功率
        self.complexity = complexity  # 生产区的复杂度（简单或复杂）

        # 创建字典来存储每个生产单元的机器数
        self.machines_count = {}

        # 初始化生产区的机器数
        self._initialize_cells()

    def _initialize_cells(self):
        """初始化生产区及其对应的机器数"""
        # 上半部分生产区
        for zone, unit_count in zip(self.work_name_up, self.unit_numbers_up):
            self.machines_count[zone] = [0] * unit_count  # 为每个生产单元初始化机器数为 0

        # 下半部分生产区
        for zone, unit_count in zip(self.work_name_down, self.unit_numbers_down):
            self.machines_count[zone] = [0] * unit_count  # 为空单元初始化机器数为 0

    def distribute_machines_randomly(self):
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
        """返回每个生产区中各单元的机器数，并存储到字典中"""
        machine_count_list = []  #列表
        for zone, units in self.machines_count.items():
            # 将每个生产区的机器数存储到字典中
            machine_count_list += units# 每个单元的机器数
            # print(
            #     f"{zone}: {', '.join([str(unit) for unit in units])} 台机器")
        merged_str = ''.join(map(str, machine_count_list))
        # 将连接后的字符串转换为整数
        merged_number = int(merged_str)
        machine_count_list_renew = merged_number
        return machine_count_list_renew

    def calculate_reduction(self, initial_time, initial_power, zone, machine_count):
        """
        根据生产区的复杂度计算每个任务的运行时间
        - 简单生产区：时间下降较大
        - 复杂生产区：时间下降较小

        参数:
        initial_time (float): 初始运行时间
        initial_power (float):初始运行功率
        zone (str): 生产区名称
        machine_count (int): 当前该生产单元的机器数量

        返回:
        float: 调整后的运行时间
        """
        if self.complexity[zone] == 'simple':
            # 简单生产区：增加机器时，运行时间下降较大（每增加1台机器，减少50%）
            reduction_factor_time = 0.50
            reduction_factor_power = 0.35

        elif self.complexity[zone] == 'complex':
            # 复杂生产区：增加机器时，运行时间下降较小（每增加1台机器，减少20%）
            reduction_factor_time = 0.20
            reduction_factor_power = 0.05

        else:
            reduction_factor_time = 0  # 默认情况下没有变化
            reduction_factor_power = 0

        return initial_time * ((1 - reduction_factor_time)**machine_count)\
              ,initial_power*((1 - reduction_factor_power)**machine_count)

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


    def generate_task(self):
        """随机生成一个任务的参数"""
        # run_time = random.randint(5, 30)  # 运行时间（秒），随机生成5到30秒之间
        # run_power = random.randint(100, 500)  # 运行功率（瓦特），随机生成100到500瓦特之间
        # sleep_time = random.randint(5, 20)  # 休眠时间（秒），随机生成5到20秒之间
        # sleep_power = random.randint(30, 100)  # 休眠功率（瓦特），随机生成30到100瓦特之间
        """为了测试NSGA2算法，我固定了任务参数"""
        run_time = 17.5  # 运行时间（秒），随机生成5到30秒之间
        run_power = 33.70  # 运行功率（瓦特），随机生成100到500瓦特之间
        sleep_time = 8.3 # 休眠时间（秒），随机生成5到20秒之间
        sleep_power = 19.1  # 休眠功率（瓦特），随机生成30到100瓦特之间
        return {'run_time': run_time, 'run_power': run_power, 'sleep_time': sleep_time, 'sleep_power': sleep_power}


    def function_1(self,sequence):#由序列改变字典，用于使用交叉变异修改机器臂分配后计算时间
        """初始化"""
        for zone, unit_count in zip(self.work_name_up, self.unit_numbers_up):
            self.machines_count[zone] = [0] * unit_count  # 为每个生产单元初始化机器数为 0

        # 下半部分生产区
        for zone, unit_count in zip(self.work_name_down, self.unit_numbers_down):
            self.machines_count[zone] = [0] * unit_count  # 为空单元初始化机器数为 0

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

        """任务的指定"""
        tasks = {}

        for zone, unit_count in zip(self.work_name_up + self.work_name_down,
                                    self.unit_numbers_up + self.unit_numbers_down):
            tasks[zone] = []
            for _ in range(unit_count):
                # 为每个单元生成随机任务列表
                tasks[zone].append([self.generate_task() for _ in range(3)])  # 每个生产单元生成3个任务

        """能量，时间计算"""
        total_energy, total_time = 0, 0
        for zone in self.work_name_up + self.work_name_down:
            for unit_index in range(len(self.machines_count[zone])):
                # 获取该生产单元的机器数量
                machine_count = self.machines_count[zone][unit_index]

                for task in tasks[zone][unit_index]:
                    run_time = task['run_time']
                    run_power = task['run_power']
                    sleep_time = task['sleep_time']
                    sleep_power = task['sleep_power']
                    run_time_renew, run_power_renew = self.calculate_reduction(run_time, run_power, zone, machine_count)

                    # 计算该任务的能量消耗
                    task_energy = self.calculate_task_energy(run_time_renew, run_power_renew, sleep_time, sleep_power) * machine_count
                    total_energy += task_energy
                    total_time += run_time_renew + sleep_time

        return total_energy, total_time


