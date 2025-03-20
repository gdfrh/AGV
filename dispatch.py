from Config import *


class Timeline:
    def __init__(self):
        self.timeline = [-3] * num_orders  # 定义时间轴，不断移动它，用-3代表最开始订单已到达生产区但是没有生产单元空闲
        self.agv_timeline = [None] * total_agv  # 小车时间轴
        self.current_time = 0  # 当前时间
        self.step = [None] * num_orders  # 定义生产单元索引，方便储存,初始为None表示没有生产单元索引
        self.agv_step = [(None, None, None, None)] * num_orders  # 定义小车全局索引，方便储存,初始为None表示没有小车索引

    def add_timeline(self, time_point, idx, unit_idx):
        # 添加时间节点
        self.timeline[idx] = time_point
        self.step[idx] = unit_idx

    def delete_timeline(self, time_point):
        # 删除时间节点,是否切换会比较好
        self.timeline.remove(time_point)

    def get_next_point(self):
        # 找到时间轴中最小的非零时间节点,-1是忙碌(在运输过程中），-2订单空闲不存在空闲小车（订单在生产单元完成了工作），-3订单空闲不存在空闲单元（小车将订单送达生产区）
        min_time_order = min(
            (time for time in self.timeline if time not in {None, 0, -1, -2, -3, float('inf')}),
            default=None)

        # 获取 list2 中所有最小时间（元组的第一位）和索引
        # 获取元组的第一位作为时间
        # 获取 list2 中所有最小时间（元组的第一位），排除为 None 和 0 的时间
        min_time_agv = min(
            (x[0] for x in self.agv_timeline if x and x[0] not in {None, 0, float('inf')}),  # 确保元素不是 None 且时间不为 0
            default=None  # 如果没有有效时间，返回 None
        )
        # 如果 min_time_agv 和 min_time_order 是有效的，可以继续处理
        if min_time_agv is not None and min_time_order is not None:
            # 获取最小时间的所有索引，首先排除 None 元素
            min_indices_agv = [i for i, tup in enumerate(self.agv_timeline) if tup is not None and tup[0] == min_time_agv]
            min_indices_order = [index for index, time in enumerate(self.timeline) if time is not None and time == min_time_order]
            self.current_time = min(min_time_agv, min_time_order)
            # 返回该时间节点的索引
            if min_time_order < min_time_agv:
                return 'order', min_indices_order
            if min_time_order > min_time_agv:
                return 'agv', min_indices_agv

        elif min_time_agv is None and min_time_order is not None:
            self.current_time = min_time_order
            min_indices_order = [index for index, time in enumerate(self.timeline) if
                                 time is not None and time == min_time_order]
            return 'order', min_indices_order

        elif min_time_order is None and min_time_agv is not None:
            self.current_time = min_time_agv
            min_indices_agv = [i for i, tup in enumerate(self.agv_timeline) if
                               tup is not None and tup[0] == min_time_agv]
            return 'agv', min_indices_agv

        if min_time_agv is None and min_time_order is None:
            return None, None
