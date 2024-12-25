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
            print(
                f"{zone}: {', '.join([str(unit) for unit in units])} 台机器")
        merged_str = ''.join(map(str, machine_count_list))
        # 将连接后的字符串转换为整数
        merged_number = int(merged_str)
        # 将合并后的数字存入列表
        machine_count_list_renew = [merged_number]
        return machine_count_list_renew

    def calculate_time_reduction(self, initial_time, zone, machine_count):
        """
        根据生产区的复杂度计算每个任务的运行时间
        - 简单生产区：时间下降较大
        - 复杂生产区：时间下降较小

        参数:
        initial_time (float): 初始运行时间
        zone (str): 生产区名称
        machine_count (int): 当前该生产单元的机器数量

        返回:
        float: 调整后的运行时间
        """
        """为了测试我固定了参数"""
        if self.complexity[zone] == 'simple':
            # 简单生产区：增加机器时，运行时间下降较大（每增加1台机器，减少30%-50%）
            # reduction_factor = random.uniform(0.3, 0.5)
            reduction_factor=0
        elif self.complexity[zone] == 'complex':
            # 复杂生产区：增加机器时，运行时间下降较小（每增加1台机器，减少10%-20%）
            # reduction_factor = random.uniform(0.1, 0.2)
            reduction_factor = 0
        else:
            reduction_factor = 0  # 默认情况下没有变化

        return initial_time * ((1 - reduction_factor)**machine_count)

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

    def calculate_unit_energy(self, zone, unit_index, tasks):
        """
        计算生产单元的总能量消耗，考虑所有任务和机器数量

        参数:
        zone (str): 生产区名称
        unit_index (int): 生产单元索引
        tasks (list): 生产单元的任务列表，每个任务是一个字典，包含任务的运行时间、运行功率、休眠时间和休眠功率

        返回:
        float: 该生产单元的总能量消耗 (焦耳, J)
        """
        # 获取该生产单元的机器数量
        machine_count = self.machines_count[zone][unit_index]
        total_unit_energy = 0

        # 计算该单元所有任务的能量消耗，同时计算多个机器臂参与时，时间的减少
        for task in tasks:
            run_time = task['run_time']
            run_time = self.calculate_time_reduction(run_time, zone, machine_count)
            run_power = task['run_power']
            sleep_time = task['sleep_time']
            sleep_power = task['sleep_power']

            # 计算该任务的能量消耗
            task_energy = self.calculate_task_energy(run_time, run_power, sleep_time, sleep_power)*machine_count

            total_unit_energy += task_energy

        return total_unit_energy

    def display_unit_energy(self, tasks):
        """显示每个生产单元的能量消耗"""
        for zone in self.work_name_up + self.work_name_down:
            print(f" {zone}")
            for unit_index in range(len(self.machines_count[zone])):
                print(f"  单元 {unit_index + 1}:")
                unit_energy = self.calculate_unit_energy(zone, unit_index, tasks[zone][unit_index])
                print(f"    总能量消耗: {unit_energy} J")

    def generate_task(self):
        """随机生成一个任务的参数"""
        # run_time = random.randint(5, 30)  # 运行时间（秒），随机生成5到30秒之间
        # run_power = random.randint(100, 500)  # 运行功率（瓦特），随机生成100到500瓦特之间
        # sleep_time = random.randint(5, 20)  # 休眠时间（秒），随机生成5到20秒之间
        # sleep_power = random.randint(30, 100)  # 休眠功率（瓦特），随机生成30到100瓦特之间
        """为了测试NSGA2算法，我固定了任务参数"""
        run_time = 20  # 运行时间（秒），随机生成5到30秒之间
        run_power = 100  # 运行功率（瓦特），随机生成100到500瓦特之间
        sleep_time = 10 # 休眠时间（秒），随机生成5到20秒之间
        sleep_power = 50  # 休眠功率（瓦特），随机生成30到100瓦特之间
        return {'run_time': run_time, 'run_power': run_power, 'sleep_time': sleep_time, 'sleep_power': sleep_power}

    def generate_tasks(self):
        """为每个生产单元生成任务"""
        tasks = {}
        for zone, unit_count in zip(self.work_name_up + self.work_name_down,
                                    self.unit_numbers_up + self.unit_numbers_down):
            tasks[zone] = []
            for _ in range(unit_count):
                # 为每个单元生成随机任务列表
                tasks[zone].append([self.generate_task() for _ in range(3)])  # 每个生产单元生成3个任务

        return tasks

    def calculate_total_energy(self, tasks):
        """计算整个车间的总能量消耗"""
        total_energy = 0
        for zone in self.work_name_up + self.work_name_down:
            for unit_index in range(len(self.machines_count[zone])):
                unit_energy = self.calculate_unit_energy(zone, unit_index, tasks[zone][unit_index])
                total_energy += unit_energy
        return total_energy

    def calculate_total_time(self, tasks):
        """计算整个车间的总时间"""
        total_time = 0
        for zone in self.work_name_up + self.work_name_down:
            for unit_index in range(len(self.machines_count[zone])):
                # 获取该生产单元的机器数量
                machine_count = self.machines_count[zone][unit_index]
                for task in tasks[zone][unit_index]:
                    run_time = task['run_time']
                    run_time_renew = self.calculate_time_reduction(run_time, zone, machine_count)
                    total_time += run_time_renew
        return total_time


    def calculate_and_display_energy(self,init_arm):
        """
        计算并显示整个车间的能量消耗，以及每个生产单元的能量消耗。

        参数:
        init_arm (Arm): 已初始化的车间对象

        返回:
        float: 整个车间的总能量消耗 (J)
        """
        # 生成任务
        tasks = init_arm.generate_tasks()

        # 显示每个生产单元的能量消耗
        init_arm.display_unit_energy(tasks)

        # 计算整个车间的总能量消耗
        total_energy = init_arm.calculate_total_energy(tasks)
        total_time = init_arm.calculate_total_time(tasks)

        # 打印总能量消耗
        print(f"\n整个车间的总能量消耗: {total_energy} J")
        print(f"\n整个车间的总时间消耗: {total_time} s")

        return total_energy, total_time

    def function_1(self,sequence):#由序列改变字典，用于使用交叉变异修改机器臂分配后计算时间
        for zone, unit_count in zip(self.work_name_up, self.unit_numbers_up):
            self.machines_count[zone] = [0] * unit_count  # 为每个生产单元初始化机器数为 0

        # 下半部分生产区
        for zone, unit_count in zip(self.work_name_down, self.unit_numbers_down):
            self.machines_count[zone] = [0] * unit_count  # 为空单元初始化机器数为 0
        sequence_idx = 0  # 追踪当前序列中的索引
        # 使用序列填充机器臂数量
        for zone in self.machines_count:
            for i in range(len(self.machines_count[zone])):
                if sequence_idx < len(list(str(sequence[0]))):
                    # 将 sequence 中的数字逐个赋值到生产区的机器臂数量
                    # 把每个元素转换为字符串，然后逐个字符赋值
                    machines = list(str(sequence[0]))  # 将数字转为字符串
                    self.machines_count[zone][i] = int(machines[sequence_idx])  # 给当前生产单元赋值机器数量
                    sequence_idx += 1


        tasks = {}
        for zone, unit_count in zip(self.work_name_up + self.work_name_down,
                                    self.unit_numbers_up + self.unit_numbers_down):
            tasks[zone] = []
            for _ in range(unit_count):
                # 为每个单元生成随机任务列表
                tasks[zone].append([self.generate_task() for _ in range(3)])  # 每个生产单元生成3个任务

        total_energy = 0
        for zone in self.work_name_up + self.work_name_down:
            for unit_index in range(len(self.machines_count[zone])):
                unit_energy = self.calculate_unit_energy(zone, unit_index, tasks[zone][unit_index])
                total_energy += unit_energy
        total_time = 0
        for zone in self.work_name_up + self.work_name_down:
            for unit_index in range(len(self.machines_count[zone])):
                # 获取该生产单元的机器数量
                machine_count = self.machines_count[zone][unit_index]
                for task in tasks[zone][unit_index]:
                    run_time = task['run_time']
                    run_time_renew = self.calculate_time_reduction(run_time, zone, machine_count)
                    total_time += run_time_renew
        return total_energy, total_time


