from Config import *


class Timeline:
    def __init__(self):
        self.timeline = [-3] * num_orders  # 定义时间轴，不断移动它，用-3代表最开始订单已到达生产区但是没有生产单元空闲
        self.agv_timeline = [None] * total_agv  # 小车时间轴
        self.current_time = 0  # 当前时间
        self.step = [0] * num_orders  # 定义生产单元索引，方便储存

    def add_timeline(self, time_point, idx, unit_idx):
        # 添加时间节点
        self.timeline[idx] = time_point
        self.step[idx] = unit_idx

    def delete_timeline(self, time_point):
        # 删除时间节点,是否切换会比较好
        self.timeline.remove(time_point)

    def get_next_point(self):
        # 找到时间轴中最小的非零时间节点,-1是忙碌(在运输过程中），-2订单空闲不存在空闲小车（订单在生产单元完成了工作），-3订单空闲不存在空闲单元（小车将订单送达生产区）
        min_time_order = min(time for time in self.timeline if time not in {None, 0, -1, -2, -3})
        # 找到该时间节点的位置
        min_indices_order = [index for index, time in enumerate(self.timeline) if time == min_time_order]

        # 获取 list2 中所有最小时间（元组的第一位）和索引
        # 获取元组的第一位作为时间
        min_time_agv = min((x[0] for x in self.agv_timeline if x[0] != 0))  # 过滤掉时间为 0 的元组
        min_indices_agv = [i for i, tup in enumerate(self.agv_timeline) if tup[0] == min_time_agv]

        self.current_time = min(min_time_agv, min_time_order)
        # 返回该时间节点的索引
        if min_time_order > min_time_agv:
            return 'agv', min_indices_agv
        if min_time_order < min_time_agv:
            return 'order', min_indices_order



