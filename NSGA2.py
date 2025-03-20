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
    dis_table = {sorted_front1[0]: np.inf, sorted_front1[-1]: np.inf, sorted_front2[0]: np.inf, sorted_front2[-1]: np.inf}
    for i in range(1, length - 1):
        k = sorted_front1[i]
        dis_table[k] = dis_table.get(k, 0)+(values1[sorted_front1[i+1]]-values1[sorted_front1[i-1]])/(max(values1)-min(values1))
    for i in range(1, length - 1):
        k = sorted_front1[i]
        dis_table[k] = dis_table[k]+(values2[sorted_front2[i+1]]-values2[sorted_front2[i-1]])/(max(values2)-min(values2))
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
    # 将列表中的数字转换为字符串
    individual1_str = str(individual1)  # 不考虑括号
    individual2_str = str(individual2)

    # # 使用正则表达式查找连续的非零数字和后面跟随的所有零
    # parts1 = re.findall(r'[1-9]+0*', individual1_str)
    # parts2 = re.findall(r'[1-9]+0*', individual2_str)
    parts1 = number_departure(individual1_str, unit_state_list1)
    parts2 = number_departure(individual2_str, unit_state_list2)

    new_parts1 = []
    new_parts2 = []

    for i in range(len(parts1)):    # 0到6
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
        idx =random.randint(0, min(idx1, idx2))
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
    if total_machines - sum_of_digits < required_machines:
        """机器臂不够新增一个生产单元，随机选择一个生产单元删除它"""
        # 随机选择一个生产单元并删除,但是得保证每一个生产区都有生产单元和机器臂
        unit_to_remove = random.choice(zone_units)
        zone_units.remove(unit_to_remove)
        if sum(zone_units) != 0:
            expanded_list = expanded_list[0:start_idx] + zone_units + expanded_list[end_idx:]
            """对应拷贝分布状态减一，避免影响之前的状态"""
            new_state[zone_index] -= 1

    elif total_machines - sum_of_digits >= required_machines:
        """机器臂可以新增一个生产单元，随机选择一个生产区，增加1个生产单元"""
        zone_units.append(required_machines)
        expanded_list = expanded_list[0:start_idx] + zone_units + expanded_list[end_idx:]
        expanded_list = sorted(expanded_list, reverse=True)
        """对应拷贝分布状态加一，避免影响之前的状态"""
        new_state[zone_index] += 1

    # 将 expanded_list 中的数字重新组合成一个新的字符串
    new_str = ''.join(map(str, expanded_list))

    total_units = sum(new_state)
    # 将组合后的字符串转换为一个整数
    new_individual = int(new_str)

    """对小车进行变异"""
    # 先随机选择一个生产区
    object_zone_agv = random.choice(work_name)
    # 找到对应的生产区的索引
    zone_index_agv = work_name.index(object_zone_agv)
    # 先记录一下选到的小车的数量
    target_agv_count = agv_count[zone_index_agv]
    if target_agv_count > 1:
        # 随机减少的数量，保证最少剩下一个小车
        reduction = random.randint(1, target_agv_count - 1)
        # 更新小车数量
        agv_count[zone_index_agv] -= reduction
        if object_zone_agv == work_name[-1]:
            agv_count[0] += reduction
        else:
            agv_count[zone_index_agv + 1] += reduction

    # 返回重新组合后的结果,是一个列表,以及分布状态
    return new_individual, new_state, agv_count

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


# NSGA2主循环
def main_loop(pop_size, max_gen, init_population,init_arm):
    gen_no = 0
    population_P = init_population.copy()
    loop_start_time = time.time()
    while gen_no <= max_gen:
        population_R = population_P.copy()
        # 根据P(t)生成Q(t),R(t)=P(t)vQ(t)
        # 计算每个解的目标函数值
        objective1 = []
        objective2 = []
        for i in range(len(population_R)):
            total_energy, total_time = init_arm.object_function_1(population_R[i], i)
            objective1.append(round(total_energy,2))  # 将 total_energy 添加到 objective1
            objective2.append(round(total_time,2))  # 将 total_time 添加到 objective2

        # 非支配排序，得到不同前沿
        fronts = fast_non_dominated_sort(objective1, objective2)
        """如果全是第一前沿呢？加判断条件"""
        # 获取非第一前沿的解，进行交叉和变异
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
        elif len(not_front_1) >= number_limits*pop_size:
            """not_front_1数量不少于x倍总种群，则使用它"""
            use_list = not_front_1
        else:
            """如果not_front_1数量太少,则全局使用"""
            use_list = front_1 + not_front_1

        # 每一次迭代之后解数量不足2 * pop_size
        while len(population_R) < 2 * pop_size:
            machine_counts1 = {}
            machine_counts2 = {}
            # 标志，若生产单元为0，则为True，反之为False
            is_zero1 = False
            is_zero2 = False
            # 随机选择2个不同的非第一前沿的个体进行交叉
            x = random.choice(use_list)
            y = random.choice(use_list)
            if x != y:#虽然应该不止一个

                new_member1, new_member2 = crossover(population_R[x], population_R[y],
                                                     init_arm.unit_states[x], init_arm.unit_states[y])
                """下一个可能的生产单元分布状态"""
                new_state1 = init_arm.unit_states[x]
                new_state2 = init_arm.unit_states[y]

                """下一个可能的小车分布数量"""
                new_agv_count1 = init_arm.agv_count[x]
                new_agv_count2 = init_arm.agv_count[y]

                # 变异操作
                if random.random() < mutation_probability:  # 变异概率
                    new_member1, new_state1,new_agv_count1= mutate(new_member1, init_arm, init_arm.unit_states[x], init_arm.agv_count[x])  # 对新个体进行变异
                    new_member2, new_state2,new_agv_count2= mutate(new_member2, init_arm, init_arm.unit_states[y], init_arm.agv_count[y])

                """首先得判断不能让某个生产区没有生产单元,我们可以使用反映射，来判断"""
                """反映射，如果生产单元个数发生了变化如何判断,我直接新设了一个字典来判断，不影响machines_count"""
                machine_counts1 = anti_mapping(new_member1, new_state1)
                # 遍历每个生产区，检查该生产区的机器数量是否全为0
                for zone, machines in machine_counts1.items():
                    total_machines_in_zone = sum(machines)
                    if total_machines_in_zone > total_machines:
                        is_zero1 = True
                    if all(machine == 0 for machine in machines):  # 如果所有机器数都为0
                        is_zero1 = True
                # 对小车:数量不能超出约束
                if sum(new_agv_count1) > total_agv:
                    is_zero1 = True

                machine_counts2 = anti_mapping(new_member2, new_state2)
                # 遍历每个生产区，检查该生产区的机器数量是否全为0
                for zone, machines in machine_counts2.items():
                    total_machines_in_zone = sum(machines)
                    if total_machines_in_zone > total_machines:
                        is_zero1 = True
                    if all(machine == 0 for machine in machines):  # 如果所有机器数都为0
                        is_zero2 = True

                # 对小车:数量不能超出约束
                if sum(new_agv_count2) > total_agv:
                    is_zero2 = True

                """不能有某个生产区机器臂为0，并且不能超过机器臂总数"""
                if not is_zero1:
                    """如果解存在了，就不考虑它"""
                    if new_member1 not in population_R:
                        population_R.append(new_member1)
                        init_arm.unit_states.append(new_state1)
                        init_arm.agv_count.append(new_agv_count1)

                if not is_zero2:
                    if new_member2 not in population_R:
                        population_R.append(new_member2)
                        init_arm.unit_states.append(new_state2)
                        init_arm.agv_count.append(new_agv_count2)

        objective1 = []
        objective2 = []
        obj_order = []

        # for i in range(2 * pop_size):
        """此时len(population_R)>=2 * pop_size"""
        for i in range(len(population_R)):
            # 通过调用 function_1，解包返回的元组（total_energy, total_time）
            total_energy, total_time, total_order = init_arm.object_function_2(population_R[i], i)

            objective1.append(round(total_energy,2))  # 将 total_energy 添加到 objective1
            objective2.append(round(total_time,2))    # 将 total_time 添加到 objective2
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

        # 得到P(t+1)重复上述过程
        population_P = population_P_next.copy()
        """新一代分布状态"""
        init_arm.unit_states = new_state.copy()
        init_arm.agv_count = new_agv_count.copy()
        if gen_no % 10 == 0:
            best_obj1 = []
            best_obj2 = []

            for i in range(pop_size):
                # 通过调用 function_1，解包返回的元组（total_energy, total_time）
                total_energy, total_time,total_order = init_arm.object_function_2(population_P[i], i)

                best_obj1.append(round(total_energy,2))  # 将 total_energy 添加到 best_obj1
                best_obj2.append(round(total_time,2))  # 将 total_time 添加到 best_obj2

            f = fast_non_dominated_sort(best_obj1, best_obj2)
            # 打印第一前沿中的目标值
            # print(f"Generation {gen_no}, first front:")
            cope_time = time.time() - loop_start_time

            print(f"Generation {gen_no}, time:{cope_time}")
            energy_pic = []
            time_pic = []
            agv_distributions = []
            order_distributions = []
            distributions_dicts = []
            for s in f[0]:
                """值可视化"""
                # print((population_P[s], 2), end=' ')
                # print()
                # print(f"Individual {s}: Energy = {best_obj1[s]}, Time = {best_obj2[s]}")
                # print('\n')
                energy_pic.append(best_obj1[s])
                time_pic.append(best_obj2[s])
                agv_distributions.append(new_agv_count[s])  # 小车分配
                order_distributions.append(obj_order_next[s])   # 订单分配
                distributions_dicts.append(anti_mapping(population_P[s], new_state[s])) # 生产单元机器臂分配（字典）
            """python图片可视化"""
            # plt.scatter(energy_pic, time_pic)
            # plt.show()
            """plotly图片可视化"""
            # 数据字典
            data = {
                'energy': energy_pic,
                'time': time_pic,
                'agv_distribution':agv_distributions,
                'order':order_distributions
            }

            # 计算最大长度
            max_length = max(len(lst) for lst in data.values())

            # 调整所有列表到相同的长度
            for key in data:
                current_length = len(data[key])
                if current_length < max_length:
                    # 补全列表
                    data[key].extend([None] * (max_length - current_length))
            # 将字典转换为 DataFrame
            df = pd.DataFrame(data)
            df_dicts = pd.DataFrame(distributions_dicts)
            # 合并 DataFrame
            df_final = pd.concat([df, df_dicts], axis=1)
            # 使用 Plotly Express 创建散点图
            fig = px.scatter(df_final, x='energy', y='time', title="Energy vs. Time Scatter Plot",
                             hover_data=['agv_distribution','组装区', '铸造区', '清洗区', '包装区','焊接区', '喷漆区', '配置区'])
            # 显示图表
            fig.show()
            loop_start_time = time.time()

        gen_no += 1

    return energy_pic, time_pic


