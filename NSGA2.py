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


def crossover(individual, idx1, idx2):
    # 确保 idx1 和 idx2 是有效的索引

    # 将列表中的数字转换为字符串
    individual_str = str(individual)#不考虑括号

    # 将字符串中的每个字符转换为整数，存入新列表
    expanded_list = [int(digit) for digit in individual_str]

    if idx1 != idx2:
        # 交换两个位置的元素
        expanded_list[idx1], expanded_list[idx2] = expanded_list[idx2], expanded_list[idx1]
    # 将 expanded_list 中的数字重新组合成一个新的字符串
    new_str = ''.join(map(str, expanded_list))

    # 将组合后的字符串转换为一个整数
    new_individual = int(new_str)

    # 返回重新组合后的结果,是一个列表
    return new_individual

def mutate(individual):
    # 将列表中的数字转换为字符串
    individual_str = str(individual)  # 不考虑括号

    # 将字符串中的每个字符转换为整数，存入新列表
    expanded_list = [int(digit) for digit in individual_str]
    # 随机的找到变异的元素的索引
    idx = random.randint(0, len(expanded_list)-1)
    # 读出这个单元分配的机器臂数量，并且任意的减少此数量，不超过总数
    n = random.randint(0,expanded_list[idx]-1)

    expanded_list[idx] -= n
    # 将 expanded_list 中的数字重新组合成一个新的字符串
    new_str = ''.join(map(str, expanded_list))

    # 将组合后的字符串转换为一个整数
    new_individual = int(new_str)

    # 返回重新组合后的结果,是一个列表
    return new_individual


# NSGA2主循环
def main_loop(pop_size, max_gen, init_population,init_arm):
    gen_no = 0
    population_P = init_population.copy()
    while gen_no < max_gen:
        population_R = population_P.copy()
        # 根据P(t)生成Q(t),R(t)=P(t)vQ(t)
        while len(population_R) != 2 * pop_size:
            x = random.randint(0, pop_size - 1)
            y = random.randint(1, len(str(population_P[x]))-2)
            idx1 = random.randint(0, y)
            idx2 = random.randint(y+1, len(str(population_P[x]))-1)
            new_member = crossover(population_P[x], idx1, idx2)
            # 变异操作
            if random.random() < 0.1:  # 假设变异概率为10%
                new_member = mutate(new_member)  # 对新个体进行变异
            if new_member not in population_R:
                population_R.append(new_member)
        # 对R(t)计算非支配前沿
        objective1 = []
        objective2 = []
        for i in range(2 * pop_size):
            # 通过调用 function_1，解包返回的元组（total_energy, total_time）
            total_energy, total_time = init_arm.function_1(population_R[i])

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
        if gen_no % 100 == 0:
            best_obj1 = []
            best_obj2 = []
            for i in range(pop_size):
                # 通过调用 function_1，解包返回的元组（total_energy, total_time）
                total_energy, total_time = init_arm.function_1(population_P[i])

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


