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

    def display_unit_energy(self, tasks):
        """显示每个生产单元的能量消耗"""
        for zone in self.work_name_up + self.work_name_down:
            print(f" {zone}")
            for unit_index in range(len(self.machines_count[zone])):
                print(f"  单元 {unit_index + 1}:")
                unit_energy = self.calculate_unit_energy(zone, unit_index, tasks[zone][unit_index])
                print(f"    总能量消耗: {unit_energy} J")

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
            run_power = task['run_power']
            run_time, run_power = self.calculate_reduction(run_time,run_power, zone, machine_count)
            sleep_time = task['sleep_time']
            sleep_power = task['sleep_power']

            # 计算该任务的能量消耗
            task_energy = self.calculate_task_energy(run_time, run_power, sleep_time, sleep_power)*machine_count

            total_unit_energy += task_energy

        return total_unit_energy