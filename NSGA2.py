import random
import numpy as np
import matplotlib.pyplot as plt
from Config import *
from robot_arm import Arm
import copy
import bisect
import time
import plotly.express as px
import pandas as pd
import re
import plotly.graph_objects as go
from scipy.stats import linregress
from scipy.optimize import curve_fit
import pickle


# 快速非支配排序
def fast_non_dominated_sort(values1, values2):
    size = len(values1)
    dominate_set = [[] for _ in range(size)]  # 解p支配的解集合
    dominated_count = [0 for _ in range(size)]  # 支配p的解数量
    solution_rank = [0 for _ in range(size)]  # 每个解的等级
    fronts = [[]]

    for p in range(size):
        dominate_set[p] = []
        dominated_count[p] = 0
        for q in range(size):
            if values1[p] <= values1[q] and values2[p] <= values2[q] \
                    and ((values1[p] == values1[q]) + (values2[p] == values2[q])) != 2:
                dominate_set[p].append(q)
            elif values1[q] <= values1[p] and values2[q] <= values2[p] \
                    and ((values1[q] == values1[p]) + (values2[q] == values2[p])) != 2:
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            solution_rank[p] = 0
            fronts[0].append(p)

    level = 0
    while fronts[level]:
        Q = []
        for p in fronts[level]:
            for q in dominate_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    solution_rank[q] = level + 1
                    if q not in Q:
                        Q.append(q)
        level = level + 1
        fronts.append(Q)
    del fronts[-1]
    return fronts


# 计算拥挤距离
def crowed_distance_assignment(values1, values2, front):
    length = len(front)
    sorted_front1 = sorted(front, key=lambda x: values1[x])
    sorted_front2 = sorted(front, key=lambda x: values2[x])
    dis_table = {sorted_front1[0]: np.inf, sorted_front1[-1]: np.inf, sorted_front2[0]: np.inf,
                 sorted_front2[-1]: np.inf}
    for i in range(1, length - 1):
        k = sorted_front1[i]
        dis_table[k] = dis_table.get(k, 0) + (values1[sorted_front1[i + 1]] - values1[sorted_front1[i - 1]]) / (
                max(values1) - min(values1))
    for i in range(1, length - 1):
        k = sorted_front1[i]
        dis_table[k] = dis_table[k] + (values2[sorted_front2[i + 1]] - values2[sorted_front2[i - 1]]) / (
                max(values2) - min(values2))
    distance = [dis_table[a] for a in front]
    return distance


def number_departure(number, splits):
    # 将数字转换为字符串
    number_str = str(number)
    # 存储结果的列表
    parts = []
    # 当前处理的起始索引
    start_index = 0
    # 遍历splits数组，按每个元素指定的数量分割字符串
    for count in splits:
        # 计算结束索引
        end_index = start_index + count
        # 检查是否超出字符串长度
        if end_index > len(number_str):
            break
        # 提取子字符串
        part = number_str[start_index:end_index]
        parts.append(part)
        # 更新起始索引为下一次迭代
        start_index = end_index
    return parts


def crossover(individual1, individual2, unit_state_list1, unit_state_list2):

    # # 使用正则表达式查找连续的非零数字和后面跟随的所有零
    # parts1 = re.findall(r'[1-9]+0*', individual1_str)
    # parts2 = re.findall(r'[1-9]+0*', individual2_str)
    parts1 = number_departure(individual1, unit_state_list1)
    parts2 = number_departure(individual2, unit_state_list2)

    new_parts1 = []
    new_parts2 = []

    for i in range(len(parts1)):  # 0到6
        # 将字符串中的每个字符转换为整数，存入新列表
        expanded1_list = [int(digit) for digit in parts1[i]]
        expanded2_list = [int(digit) for digit in parts2[i]]
        """如果选择的2个位置都是0或者是相同的数字，那么交换之后解是不变的，我们可以在之后进行判断"""
        # 交换两个位置的元素
        # 找出两个列表中非零元素的索引
        non_zero_indices1 = [i for i, x in enumerate(expanded1_list) if x != 0]
        non_zero_indices2 = [i for i, x in enumerate(expanded2_list) if x != 0]
        idx1 = random.choice(non_zero_indices1)
        idx2 = random.choice(non_zero_indices2)
        idx = random.randint(0, min(idx1, idx2))
        expanded1_list[idx], expanded2_list[idx] = expanded2_list[idx], expanded1_list[idx]

        # 排序列表，从大到小
        expanded1_list = sorted(expanded1_list, reverse=True)
        expanded2_list = sorted(expanded2_list, reverse=True)

        # 将 expanded_list 中的数字重新组合成一个新的字符串
        new_str1 = ''.join(map(str, expanded1_list))
        new_str2 = ''.join(map(str, expanded2_list))

        # 添加到新列表
        new_parts1.append(new_str1)
        new_parts2.append(new_str2)
    new_parts1 = ''.join(map(str, new_parts1))
    new_parts2 = ''.join(map(str, new_parts2))

    # # 将组合后的字符串转换为一个整数
    """前导0？如果从大到小排序，对应交换，第一个就不会是零,否则规定一下位数"""
    new_individual1 = int(new_parts1)
    new_individual2 = int(new_parts2)

    """不能超过总机器臂数量，否则重新交叉，我觉得不需要在这里管这个约束，直接在变异之后再判断就行"""
    # if sum1 <= total_machines and sum2 <= total_machines:
    # 返回重新组合后的结果,是一个数字
    return new_individual1, new_individual2


def mutate(individual, init_arm, unit_state, agv_count):
    # 将列表中的数字转换为字符串
    individual_str = str(individual)  # 不考虑括号

    # 将字符串中的每个字符转换为整数，存入新列表
    expanded_list = [int(digit) for digit in individual_str]
    # 计算 individual 中每一位的值的和
    sum_of_digits = sum(expanded_list)

    """随机找到一个生产区，来对它进行变异，我还得得知之前的生产单元的分布列表"""
    # 先随机选择一个生产区
    object_zone = random.choice(work_name)
    # 找到对应的生产区的索引
    zone_index = work_name.index(object_zone)
    # 找到对应与individual序列的起始位置和终点位置
    start_idx = sum(unit_state[:zone_index])
    end_idx = start_idx + unit_state[zone_index]
    # 获取该生产区在expanded_list中的所有生产单元
    zone_units = expanded_list[start_idx:end_idx]
    required_machines = 0
    # 遍历 zone_requirements 列表查找
    for zone_name, machines in zone_requirements:
        if zone_name == object_zone:
            required_machines = machines
            break

    """新的分布状态暂时和交叉的时候的分布状态一致"""
    new_state = copy.deepcopy(unit_state)
    # 增加或减少机器臂数量
    if random.random() < 0.5:  # 50%的概率选择增减机器臂
        # 随机选择一个生产单元，并增加机器臂
        max_possible_machines = total_machines - sum_of_digits  # 最大可用机器臂
        # 如果机器臂数量还可以增加
        if max_possible_machines > 0:
            for unit in range(len(zone_units)):
                if zone_units[unit] != 0:  # 增加机器臂
                    zone_units[unit] += 1
        else:
            for unit in range(len(zone_units)):
                if zone_units[unit] >= required_machines + 1:  # 减少机器臂，但是不能少于需要的机器臂
                    zone_units[unit] -= 1
        zone_units = sorted(zone_units, reverse=True)
        expanded_list = expanded_list[0:start_idx] + zone_units + expanded_list[end_idx:]
    else:
        if total_machines - sum_of_digits < required_machines:
            """机器臂不够新增一个生产单元，随机选择一个生产单元删除它"""
            # 随机选择一个生产单元并删除,但是得保证每一个生产区都有生产单元和机器臂
            # 随机选择一个生产单元并将其机器臂数设为0
            unit_to_remove_idx = random.choice([i for i, unit in enumerate(zone_units) if unit > 0])  # 只选择有机器臂的生产单元
            zone_units[unit_to_remove_idx] = 0  # 将选择的生产单元的机器臂数设为0
            zone_units = sorted(zone_units, reverse=True)
            if sum(zone_units) != 0:
                expanded_list = expanded_list[0:start_idx] + zone_units + expanded_list[end_idx:]
                # """对应拷贝分布状态减一，避免影响之前的状态"""
                # new_state[zone_index] -= 1

        elif total_machines - sum_of_digits >= required_machines:
            """机器臂可以新增一个生产单元，随机选择一个生产区，增加1个生产单元"""
            unit_to_add_idx = random.choice([i for i, unit in enumerate(zone_units) if unit == 0])  # 只选择有机器臂的生产单元
            if unit_to_add_idx:
                zone_units[unit_to_add_idx] = required_machines
                zone_units = sorted(zone_units, reverse=True)
                expanded_list = expanded_list[0:start_idx] + zone_units + expanded_list[end_idx:]
            # expanded_list = sorted(expanded_list, reverse=True)
            # """对应拷贝分布状态加一，避免影响之前的状态"""
            # new_state[zone_index] += 1

    # 将 expanded_list 中的数字重新组合成一个新的字符串
    new_str = ''.join(map(str, expanded_list))

    total_units = sum(new_state)
    # 将组合后的字符串转换为一个整数
    new_individual = int(new_str)

    """对小车进行变异"""
    """我现在改成随机分配，感觉效果好一些"""
    # 小车分布
    new_agv_count = agv_count[:]
    total_agv_number = total_agv
    # 将每个生产单元的机器臂数量设为0
    new_agv_count = [1] * len(new_agv_count)
    remaining_agv_number = total_agv_number - sum(new_agv_count)
    # 随机分配剩余的小车数量
    while remaining_agv_number > 0:
        # 随机选择一个生产单元，并增加一个小车
        random_index = random.choice(range(len(new_agv_count)))
        new_agv_count[random_index] += 1
        remaining_agv_number -= 1
    # # 先随机选择一个生产区
    # object_zone_agv = random.choice(work_name)
    # # 找到对应的生产区的索引
    # zone_index_agv = work_name.index(object_zone_agv)
    # # 先记录一下选到的小车的数量
    # target_agv_count = agv_count[zone_index_agv]
    # if target_agv_count > 1:
    #     # 随机减少的数量，保证最少剩下一个小车
    #     reduction = random.randint(1, target_agv_count - 1)
    #     # 更新小车数量
    #     agv_count[zone_index_agv] -= reduction
    #     if object_zone_agv == work_name[-1]:
    #         agv_count[0] += reduction
    #     else:
    #         agv_count[zone_index_agv + 1] += reduction

    # 返回重新组合后的结果,是一个列表,以及分布状态
    return new_individual, new_state, new_agv_count


def anti_mapping(sequence, unit_state):
    """反映射:将序列填充到 machines_count 内部"""
    sequence_idx = 0  # 追踪当前序列中的索引
    new_machines_count = {}  # 存储反映射后的机器臂数量
    idx = 0  # 单元数量的索引

    # 遍历每个生产区
    for zone in work_name:
        # 获取当前生产区的生产单元数
        num_units = unit_state[idx]
        new_machines_count[zone] = [0] * num_units  # 初始化该生产区的机器臂列表
        idx += 1
        # 将 `sequence` 中的数字逐个赋值到生产单元
        for i in range(num_units):
            if sequence_idx < len(list(str(sequence))):  # 防止索引越界
                machines = list(str(sequence))  # 将数字转为字符串
                new_machines_count[zone][i] = int(machines[sequence_idx])  # 将序列中的值分配给当前生产单元
                sequence_idx += 1

    return new_machines_count


def select_front(fronts):
    # 选择使用的前沿
    not_front_1 = []  # 非第一前沿的解的索引
    for level in range(1, len(fronts)):  # 从第二前沿开始
        for s in fronts[level]:
            not_front_1.append(s)
    front_1 = []  # 第一前沿的解的索引
    for s in fronts[0]:  # 从第一前沿开始
        front_1.append(s)
    if not not_front_1:
        """not_front_1为空，则使用第一前沿解"""
        use_list = front_1
    elif len(not_front_1) >= number_limits * pop_size:
        """not_front_1数量不少于x倍总种群，则使用它"""
        use_list = not_front_1
    else:
        """如果not_front_1数量太少,则全局使用"""
        use_list = front_1 + not_front_1
    return use_list


def crossover_and_mutation(use_list, population_R, init_arm):
    # 随机选择2个不同的非第一前沿的个体进行交叉
    x, y = random.sample(use_list, 2)

    new_member1, new_member2 = crossover(population_R[x], population_R[y],
                                         init_arm.unit_states[x], init_arm.unit_states[y])
    """下一个可能的生产单元分布状态"""
    new_state1 = init_arm.unit_states[x][:]
    new_state2 = init_arm.unit_states[y][:]

    """下一个可能的小车分布数量"""
    new_agv_count1 = init_arm.agv_count[x][:]
    new_agv_count2 = init_arm.agv_count[y][:]

    """下一个可能的订单排列"""
    new_order1 = init_arm.orders_list[x][:]
    new_order2 = init_arm.orders_list[y][:]
    # 变异操作
    if random.random() < mutation_probability:  # 变异概率
        new_member1, new_state1, new_agv_count1 = mutate(new_member1, init_arm, init_arm.unit_states[x],
                                                         init_arm.agv_count[x])  # 对新个体进行变异
        new_member2, new_state2, new_agv_count2 = mutate(new_member2, init_arm, init_arm.unit_states[y],
                                                         init_arm.agv_count[y])

    return new_member1, new_member2, new_state1, new_state2, new_agv_count1, new_agv_count2, new_order1, new_order2


def check_and_add_solution(new_member, new_state, new_agv_count, new_order, population_R, init_arm, total_machines, total_agv):
    """检查并添加解，如果满足条件则添加到种群中"""
    # 反映射，检查生产单元的机器数量
    machine_counts = anti_mapping(new_member, new_state)
    is_zero = False

    # 遍历每个生产区，检查该生产区的机器数量是否全为0
    for zone, machines in machine_counts.items():
        total_machines_in_zone = sum(machines)
        if total_machines_in_zone > total_machines:
            is_zero = True
        if all(machine == 0 for machine in machines):  # 如果所有机器数都为0
            is_zero = True

    # 对小车:数量不能超出约束
    if sum(new_agv_count) > total_agv:
        is_zero = True

    # 如果解有效，且不在种群中，添加到种群中
    if not is_zero and new_member not in population_R:
        population_R.append(new_member)
        init_arm.unit_states.append(new_state)
        init_arm.agv_count.append(new_agv_count)
        init_arm.orders_list.append(new_order)

def cope_with_random_solution(population_R, init_arm):
    # 定义一个新字典
    arm_distributions = {}
    # 随机选择一个部署解
    idx = random.choice(range(len(population_R)))
    for zone, unit_count in zip(work_name, init_arm.unit_numbers):
        arm_distributions[zone] = [0] * unit_count  # 为每个生产单元初始化机器数为 0
        # 1. 根据需求给每个生产区分配机器
    remaining_machines = total_machines  # 总机器数量
    for zone, min_machines in zone_requirements:  # zone_requirements 中保存每个生产区需要的最小机器数量
        # 获取该生产区的单元数
        units = len(arm_distributions[zone])
        obj_units = random.randint(0, units - 1)
        # 先给每个生产单元分配最低机器臂数量
        arm_distributions[zone][obj_units] = min_machines
        remaining_machines -= min_machines
    # 2. 随机分配机器
    while remaining_machines > 0:
        zone = random.choice(work_name)  # 随机选择一个生产区
        unit_index = random.randint(0, len(arm_distributions[zone]) - 1)  # 随机选择一个生产单元
        # 分配每个生产区需要的机器数，而不是逐个机器分配
        """如果机器臂数量为0，则分配最低要求数"""
        if arm_distributions[zone][unit_index] == 0:
            min_required_machines = zone_requirements[work_name.index(zone)][1]  # 获取该生产区的最小机器需求
            if remaining_machines >= min_required_machines:
                arm_distributions[zone][unit_index] = min_required_machines
                remaining_machines -= min_required_machines
            """如果已分配机器臂，就多分配1个"""
        elif arm_distributions[zone][unit_index] != 0:
            arm_distributions[zone][unit_index] += 1
            remaining_machines -= 1
    # 3. 对每个生产区的生产单元按机器数量从大到小排序
    for zone in arm_distributions:
        # 排序每个生产区的单元，按机器数量从大到小
        arm_distributions[zone] = sorted(arm_distributions[zone], reverse=True)
    machine_count_list = []  # 列表
    for zone, units in arm_distributions.items():
        # 将每个生产区的机器数存储到字典中
        machine_count_list += units  # 每个单元的机器数
    merged_str = ''.join(map(str, machine_count_list))
    # 将连接后的字符串转换为整数
    merged_number = int(merged_str)
    unit_state = init_arm.unit_states[idx][:]

    # 小车分布
    agv_count = init_arm.agv_count[idx][:]
    total_agv_number = total_agv
    # 将每个生产单元的机器臂数量设为0
    agv_count = [1] * len(agv_count)
    remaining_agv_number = total_agv_number - sum(agv_count)
    # 随机分配剩余的小车数量
    while remaining_agv_number > 0:
        # 随机选择一个生产单元，并增加一个机器臂
        random_index = random.choice(range(len(agv_count)))
        agv_count[random_index] += 1
        remaining_agv_number -= 1
    if merged_number not in population_R:
        population_R.append(merged_number)
        init_arm.unit_states.append(unit_state)
        init_arm.agv_count.append(agv_count)
        init_arm.orders_list.append(init_arm.orders_list[idx])

# NSGA2主循环
def main_loop(pop_size, max_gen, init_population, init_arm):
    gen_no = 0
    best_solution_1 = []
    best_solution_2 = []  # 用于存储每一代的最优解
    best_solutions_info = []  # 用于存储最优解对应的分布信息
    population_P = init_population.copy()
    loop_start_time = time.time()
    while gen_no <= max_gen:
        population_R = population_P.copy()
        # 根据P(t)生成Q(t),R(t)=P(t)vQ(t)
        # 计算每个解的目标函数值
        objective1 = []
        objective2 = []
        if compare in (0, 1, 2):
            for i in range(len(population_R)):
                # print(init_arm.orders_list[i])
                total_energy, total_time = init_arm.object_function_1(population_R[i], i)
                objective1.append(round(total_energy, 2))  # 将 total_energy 添加到 objective1
                objective2.append(round(total_time, 2))  # 将 total_time 添加到 objective2

            # 非支配排序，得到不同前沿
            fronts = fast_non_dominated_sort(objective1, objective2)
            """如果全是第一前沿呢？加判断条件"""
            # 获取非第一前沿的解，进行交叉和变异
            use_list = select_front(fronts)
            # 每一次迭代之后解数量不足2 * pop_size
            while len(population_R) < 2 * pop_size:
                # 随机选择2个不同的非第一前沿的个体进行交叉、变异
                new_member1, new_member2, new_state1, new_state2, new_agv_count1, new_agv_count2, new_order1, new_order2 = (
                                                        crossover_and_mutation(use_list, population_R, init_arm))
                # 检查是否满足约束
                check_and_add_solution(new_member1, new_state1, new_agv_count1, new_order1, population_R, init_arm, total_machines, total_agv)
                check_and_add_solution(new_member2, new_state2, new_agv_count2, new_order2, population_R, init_arm, total_machines, total_agv)
        if compare in (3, 4):
            while len(population_R) < 2 * pop_size:
                cope_with_random_solution(population_R,init_arm)

        objective1 = []
        objective2 = []
        obj_order = []

        # for i in range(2 * pop_size):
        """此时len(population_R)>=2 * pop_size"""
        for i in range(len(population_R)):
            # 通过调用 function_1，解包返回的元组（total_energy, total_time）
            total_energy, total_time, total_order = init_arm.object_function_compare(population_R[i], i)
            objective1.append(round(total_energy, 2))  # 将 total_energy 添加到 objective1
            objective2.append(round(total_time, 2))  # 将 total_time 添加到 objective2
            obj_order.append(total_order)
        fronts = fast_non_dominated_sort(objective1, objective2)

        # 获取P(t+1)，先从等级高的fronts复制，然后在同一层front根据拥挤距离选择
        population_P_next = []
        choose_solution = []
        obj_order_next = []
        # 存储新一代的分布状态
        new_state = []
        new_agv_count = []

        level = 0
        while len(population_P_next) + len(fronts[level]) <= pop_size:
            for s in fronts[level]:
                choose_solution.append(s)
                new_state.append(init_arm.unit_states[s])
                new_agv_count.append(init_arm.agv_count[s])
                population_P_next.append(population_R[s])
                obj_order_next.append(obj_order[s])
            level += 1
        if len(population_P_next) != pop_size:
            level_distance = crowed_distance_assignment(objective1, objective2, fronts[level])
            sort_solution = sorted(fronts[level], key=lambda x: level_distance[fronts[level].index(x)], reverse=True)
            for i in range(pop_size - len(population_P_next)):
                choose_solution.append(sort_solution[i])
                new_state.append(init_arm.unit_states[sort_solution[i]])
                new_agv_count.append(init_arm.agv_count[sort_solution[i]])
                population_P_next.append(population_R[sort_solution[i]])
                obj_order_next.append(obj_order[sort_solution[i]])

        # — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —
        """保留最优解"""
        for i in fronts[0]:
            # 获取最优解对应的生产单元分布、订单分布和小车分布
            best_solution_1.append(objective1[i])
            best_solution_2.append(objective2[i])
            best_solution_info = {
                'distributions': anti_mapping(population_R[i], init_arm.unit_states[i]),  # 生产单元分配
                'agv_count': init_arm.agv_count[i],  # 小车分布
                'orders_list': init_arm.orders_list[i],  # 订单顺序
                'timeline_history':init_arm.timeline_history[i],  # 订单时间节点记录
                'agv_timeline_history':init_arm.agv_timeline_history[i]  # 小车时间节点记录
            }
            # 保存最优解和其分布信息
            best_solutions_info.append(best_solution_info)
        # — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —
        # 得到P(t+1)重复上述过程
        population_P = population_P_next.copy()
        """新一代分布状态"""
        init_arm.unit_states = new_state.copy()
        init_arm.agv_count = new_agv_count.copy()
        init_arm.orders_list = obj_order_next.copy()

        if gen_no % 10 == 0:
            cope_time = time.time() - loop_start_time
            print(f"Generation {gen_no}, time:{cope_time}")

        if gen_no == max_gen:
            # 从历史最优解中取得全局最优解
            fronts = fast_non_dominated_sort(best_solution_1, best_solution_2)

            energy_pic = []
            time_pic = []
            agv_distributions = []
            order_distributions = []
            distributions_dicts = []
            timeline_history = []
            agv_timeline_history = []
            for s in fronts[0]:
                energy_pic.append(best_solution_1[s])
                time_pic.append(best_solution_2[s])
                agv_distributions.append(best_solutions_info[s]['agv_count'])  # 小车分配
                order_distributions.append(best_solutions_info[s]['orders_list'])  # 订单分配
                distributions_dicts.append(best_solutions_info[s]['distributions'])  # 生产单元机器臂分配（字典）
                timeline_history.append(best_solutions_info[s]['timeline_history'])
                agv_timeline_history.append(best_solutions_info[s]['agv_timeline_history'])
            # 将数据保存到文件
            with open('timeline_history.pkl', 'wb') as file:
                pickle.dump(timeline_history[0], file)

            with open('agv_timeline_history.pkl', 'wb') as file:
                pickle.dump(agv_timeline_history[0], file)

                # 数据字典
            data = {
                'energy': energy_pic,
                'time': time_pic,
                'agv_distribution': agv_distributions,
                'order': order_distributions,
                'distributions_dict': distributions_dicts
            }
            # 存储在文件中，之后一起进行绘制
            with open(f'{compare}.pkl', 'wb') as file:
                pickle.dump(data, file)
            # 存储在文件中，之后一起进行绘制
            if compare == 0:
                with open(f'num_orders{num_orders}.pkl', 'wb') as file:
                    pickle.dump(data, file)
                with open(f'total_machines{total_machines}.pkl', 'wb') as file:
                    pickle.dump(data, file)
            # # 计算最大长度
            # max_length = max(len(lst) for lst in data.values())
            #
            # # 调整所有列表到相同的长度
            # for key in data:
            #     current_length = len(data[key])
            #     if current_length < max_length:
            #         # 补全列表
            #         data[key].extend([None] * (max_length - current_length))
            # # 将字典转换为 DataFrame
            # df = pd.DataFrame(data)
            # df_dicts = pd.DataFrame(distributions_dicts)
            # # 合并 DataFrame
            # df_final = pd.concat([df, df_dicts], axis=1)
            # # 使用 Plotly Express 创建散点图
            # fig = px.scatter(df_final, x='energy', y='time', title="Energy vs. Time Scatter Plot",
            #                  hover_data=['agv_distribution', '组装区', '铸造区', '清洗区', '包装区', '焊接区', '喷漆区',
            #                              '配置区'])

            # # 获取 energy 和 time 数据
            # x_data = df_final['energy'].dropna()  # 删除 NaN 值
            # y_data = df_final['time'].dropna()  # 删除 NaN 值
            #
            # # 定义反比例函数模型
            # def inverse_model(x, a, b):
            #     return a / (x + b)
            #
            # # 使用 curve_fit 进行反比例函数拟合
            # params, params_covariance = curve_fit(inverse_model, x_data, y_data, p0=[1, 1])
            # # 拟合的参数 a 和 b
            # a, b = params
            #
            # # 创建拟合曲线
            # x_fit = np.linspace(min(x_data), max(x_data), 100)
            # y_fit = inverse_model(x_fit, *params)
            #
            # # 将拟合曲线添加到图中
            # fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Inverse Fit Line', line=dict(color='red')))

            # 显示图表
            # fig.show()

        gen_no += 1

        # if gen_no % 10 ==0:
        #     best_obj1 = []
        #     best_obj2 = []
        #
        #     for i in range(pop_size):
        #         # 通过调用 function_1，解包返回的元组（total_energy, total_time）
        #         total_energy, total_time,total_order = init_arm.object_function_2(population_P[i], i)
        #
        #         best_obj1.append(round(total_energy,2))  # 将 total_energy 添加到 best_obj1
        #         best_obj2.append(round(total_time,2))  # 将 total_time 添加到 best_obj2
        #
        #     f = fast_non_dominated_sort(best_obj1, best_obj2)
        #     # 打印第一前沿中的目标值
        #     # print(f"Generation {gen_no}, first front:")
        #     cope_time = time.time() - loop_start_time
        #
        #     print(f"Generation {gen_no}, time:{cope_time}")
        #     energy_pic = []
        #     time_pic = []
        #     agv_distributions = []
        #     order_distributions = []
        #     distributions_dicts = []
        #     for s in f[0]:
        #         """值可视化"""
        #         # print((population_P[s], 2), end=' ')
        #         # print()
        #         # print(f"Individual {s}: Energy = {best_obj1[s]}, Time = {best_obj2[s]}")
        #         # print('\n')
        #         energy_pic.append(best_obj1[s])
        #         time_pic.append(best_obj2[s])
        #         agv_distributions.append(new_agv_count[s])  # 小车分配
        #         order_distributions.append(obj_order_next[s])   # 订单分配
        #         distributions_dicts.append(anti_mapping(population_P[s], new_state[s])) # 生产单元机器臂分配（字典）
        #     """python图片可视化"""
        #     # plt.scatter(energy_pic, time_pic)
        #     # plt.show()
        #     """plotly图片可视化"""

    #         # 数据字典
    #         data = {
    #             'energy': energy_pic,
    #             'time': time_pic,
    #             'agv_distribution':agv_distributions,
    #             'order':order_distributions
    #         }
    #
    #         # 计算最大长度
    #         max_length = max(len(lst) for lst in data.values())
    #
    #         # 调整所有列表到相同的长度
    #         for key in data:
    #             current_length = len(data[key])
    #             if current_length < max_length:
    #                 # 补全列表
    #                 data[key].extend([None] * (max_length - current_length))
    #         # 将字典转换为 DataFrame
    #         df = pd.DataFrame(data)
    #         df_dicts = pd.DataFrame(distributions_dicts)
    #         # 合并 DataFrame
    #         df_final = pd.concat([df, df_dicts], axis=1)
    #         # 使用 Plotly Express 创建散点图
    #         fig = px.scatter(df_final, x='energy', y='time', title="Energy vs. Time Scatter Plot",
    #                          hover_data=['agv_distribution','组装区', '铸造区', '清洗区', '包装区','焊接区', '喷漆区', '配置区'])
    #
    #         # 获取 energy 和 time 数据
    #         x_data = df_final['energy'].dropna()  # 删除 NaN 值
    #         y_data = df_final['time'].dropna()  # 删除 NaN 值
    #
    #         # 定义反比例函数模型
    #         def inverse_model(x, a, b):
    #             return a / (x + b)
    #
    #         # 使用 curve_fit 进行反比例函数拟合
    #         params, params_covariance = curve_fit(inverse_model, x_data, y_data, p0=[1, 1])
    #         # 拟合的参数 a 和 b
    #         a, b = params
    #
    #         # 创建拟合曲线
    #         x_fit = np.linspace(min(x_data), max(x_data), 100)
    #         y_fit = inverse_model(x_fit, *params)
    #
    #         # 将拟合曲线添加到图中
    #         fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Inverse Fit Line', line=dict(color='red')))
    #
    #         # 显示图表
    #         fig.show()
    #         loop_start_time = time.time()
    #
    #     gen_no += 1
    #
    # return energy_pic, time_pic
