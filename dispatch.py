from Config import *


class Timeline:
    def __init__(self):
        self.timeline = [0] * num_orders  # 定义时间轴，不断移动它
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
        # 到达该时间节点
        self.current_time = min(time for time in self.timeline if time != 0)
        return self.timeline.index(self.current_time)
