import random
import numpy as np
import matplotlib.pyplot as plt
from Config import *
from robot_arm import Arm


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


def crossover(individual1, individual2, idx, total_units):
    # 确保 individual1 和 individual2 不是同一个解

    # 将列表中的数字转换为字符串
    individual1_str = str(individual1)  # 不考虑括号
    individual2_str = str(individual2)

    # 将字符串中的每个字符转换为整数，存入新列表
    expanded1_list = [int(digit) for digit in individual1_str]
    expanded2_list = [int(digit) for digit in individual2_str]
    """如果选择的2个位置都是0或者是相同的数字，那么交换之后解是不变的，我们可以在之后进行判断"""
    # 交换两个位置的元素
    expanded1_list[idx], expanded2_list[idx] = expanded2_list[idx], expanded1_list[idx]
    # 将 expanded_list 中的数字重新组合成一个新的字符串
    new_str1 = ''.join(map(str, expanded1_list))
    new_str2 = ''.join(map(str, expanded2_list))

    # 将组合后的字符串转换为一个整数
    """前导0？如果从大到小排序，对应交换，第一个就不会是零,否则规定一下位数"""
    new_individual1 = int(new_str1)
    new_individual2 = int(new_str2)
    new_individual1 = f"{new_individual1:0{total_units}d}"
    new_individual2 = f"{new_individual2:0{total_units}d}"

    # 返回重新组合后的结果,是一个数字
    return new_individual1, new_individual2

def mutate(individual, total_units):
    # 将列表中的数字转换为字符串
    individual_str = str(individual)  # 不考虑括号

    # 将字符串中的每个字符转换为整数，存入新列表
    expanded_list = [int(digit) for digit in individual_str]
    # 随机找到变异的元素的索引
    idx = random.randint(0, len(expanded_list)-1)
    # 读出这个单元分配的机器臂数量，并且任意减少此数量，不超过总数
    """还没考虑加，可以先sum来判断是否满足约束，如果等于总数，变异只考虑减法，如果小于总数概率考虑加减
    我考虑的是，若剩余机器臂小于最小所需，就随机改变已有生产单元机器臂，大于就增加生产单元并拥有最小机器臂数"""
    if expanded_list[idx] >= 1:
        n = random.randint(0, expanded_list[idx]-1)
        expanded_list[idx] -= n
    # 将 expanded_list 中的数字重新组合成一个新的字符串
    new_str = ''.join(map(str, expanded_list))

    # 将组合后的字符串转换为一个整数
    new_individual = int(new_str)
    new_individual = f"{new_individual:0{total_units}d}"

    # 返回重新组合后的结果,是一个列表
    return new_individual

def anti_mapping(init_arm,sequence):
    """反映射:将序列填充到machines_count内部"""
    sequence_idx = 0  # 追踪当前序列中的索引
    # 使用序列填充机器臂数量
    for zone in init_arm.machines_count:
        for i in range(len(init_arm.machines_count[zone])):
            if sequence_idx < len(list(str(sequence))):
                # 将 sequence 中的数字逐个赋值到生产区的机器臂数量
                # 把每个元素转换为字符串，然后逐个字符赋值
                machines = list(str(sequence))  # 将数字转为字符串
                init_arm.machines_count[zone][i] = int(machines[sequence_idx])  # 给当前生产单元赋值机器数量
                sequence_idx += 1
    return init_arm.machines_count

# NSGA2主循环
def main_loop(pop_size, max_gen, init_population,init_arm):
    gen_no = 0
    population_P = init_population.copy()
    while gen_no <= max_gen:
        population_R = population_P.copy()
        # 根据P(t)生成Q(t),R(t)=P(t)vQ(t)
        # while len(population_R) != 2 * pop_size:
        #     x = random.randint(0, pop_size - 1)
        #     y = random.randint(1, len(str(population_P[x]))-2)
        #     idx1 = random.randint(0, y)
        #     idx2 = random.randint(y+1, len(str(population_P[x]))-1)
        #     new_member = crossover(population_P[x], idx1, idx2)
        #     # 变异操作
        #     if random.random() < 0.1:  # 假设变异概率为10%
        #         new_member = mutate(new_member)  # 对新个体进行变异
        #     if new_member not in population_R:
        #         population_R.append(new_member)
        # 对R(t)计算非支配前沿
        # 计算每个解的目标函数值
        objective1 = []
        objective2 = []
        for i in range(len(population_R)):
            total_energy, total_time = init_arm.object_function(population_R[i])
            objective1.append(total_energy)  # 将 total_energy 添加到 objective1
            objective2.append(total_time)  # 将 total_time 添加到 objective2

        # 非支配排序，得到不同前沿
        fronts = fast_non_dominated_sort(objective1, objective2)
        """如果全是第一前沿呢？加判断条件"""
        # 获取非第一前沿的解，进行交叉和变异
        non_front_1 = []  # 非第一前沿的解的索引
        for level in range(1, len(fronts)):  # 从第二前沿开始
            for s in fronts[level]:
                non_front_1.append(s)
        front_1 = []  # 第一前沿的解的索引
        for s in fronts[0]:  # 从第一前沿开始
                front_1.append(s)
        if non_front_1:
            """non_front_1不为空，则使用它"""
            use_list = non_front_1
        else:
            """non_front_1为空，则使用第一前沿解"""
            use_list = front_1
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
            if x != y:
                # 再找到交叉的位置
                idx = random.randint(0, len(str(population_R[x])) - 1)
                """如果变异改变生产单元数，这里是否考虑修改？"""
                new_member1, new_member2 = crossover(population_R[x], population_R[y], idx, sum(init_arm.unit_numbers))

                # 变异操作
                if random.random() < 0.2:  # 假设变异概率为20%
                    new_member1 = mutate(new_member1, sum(init_arm.unit_numbers))  # 对新个体进行变异
                    new_member2 = mutate(new_member2, sum(init_arm.unit_numbers))

                """如果选择的2个位置都是0或者是相同的数字，那么交换之后解是不变的,如果解存在了，就不考虑它"""
                """首先得判断不能让某个生产区没有生产单元,我们可以使用反映射，来判断"""

                """反映射"""
                machine_counts1 = anti_mapping(init_arm, new_member1)
                # 遍历每个生产区，检查该生产区的机器数量是否全为0
                for zone, machines in machine_counts1.items():
                    if all(machine == 0 for machine in machines):  # 如果所有机器数都为0
                        is_zero1 = True

                machine_counts2 = anti_mapping(init_arm, new_member2)
                # 遍历每个生产区，检查该生产区的机器数量是否全为0
                for zone, machines in machine_counts2.items():
                    if all(machine == 0 for machine in machines):  # 如果所有机器数都为0
                        is_zero2 = True

                if not is_zero1:
                    if new_member1 not in population_R:
                        population_R.append(new_member1)

                if not is_zero2:
                    if new_member2 not in population_R:
                        population_R.append(new_member2)

        objective1 = []
        objective2 = []
        for i in range(2 * pop_size):
            # 通过调用 function_1，解包返回的元组（total_energy, total_time）
            total_energy, total_time = init_arm.object_function(population_R[i])

            objective1.append(total_energy)  # 将 total_energy 添加到 objective1
            objective2.append(total_time)    # 将 total_time 添加到 objective2
        fronts = fast_non_dominated_sort(objective1, objective2)
        # 获取P(t+1)，先从等级高的fronts复制，然后在同一层front根据拥挤距离选择
        population_P_next = []
        choose_solution = []
        level = 0
        while len(population_P_next) + len(fronts[level]) <= pop_size:
            for s in fronts[level]:
                choose_solution.append(s)
                population_P_next.append(population_R[s])
            level += 1
        if len(population_P_next) != pop_size:
            level_distance = crowed_distance_assignment(objective1, objective2, fronts[level])
            sort_solution = sorted(fronts[level], key=lambda x: level_distance[fronts[level].index(x)], reverse=True)
            for i in range(pop_size - len(population_P_next)):
                choose_solution.append(sort_solution[i])
                population_P_next.append(population_R[sort_solution[i]])
        # 得到P(t+1)重复上述过程
        population_P = population_P_next.copy()
        if gen_no % 2 == 0:
            best_obj1 = []
            best_obj2 = []
            for i in range(pop_size):
                # 通过调用 function_1，解包返回的元组（total_energy, total_time）
                total_energy, total_time = init_arm.object_function(population_P[i])

                best_obj1.append(total_energy)  # 将 total_energy 添加到 best_obj1
                best_obj2.append(total_time)  # 将 total_time 添加到 best_obj2
            f = fast_non_dominated_sort(best_obj1, best_obj2)
            # 打印第一前沿中的目标值
            print(f"Generation {gen_no}, first front:")
            energy_pic=[]
            time_pic=[]
            for s in f[0]:
                print((population_P[s], 2), end=' ')
                print()
                print(f"Individual {s}: Energy = {best_obj1[s]}, Time = {best_obj2[s]}")
                print('\n')
                energy_pic.append(best_obj1[s])
                time_pic.append(best_obj2[s])
            plt.scatter(energy_pic, time_pic)
            plt.show()
            """我感觉是点覆盖了，结果重复，所以需要对时间和功率进行调整
            至于s的问题，应该是之前排序过了，所以最优的已经往前放了，就会顺序执行"""
        gen_no += 1

    return energy_pic, time_pic


