class Timeline:
    def __init__(self, step):
        self.timeline = []  # 定义时间轴，不断移动它
        self.current_time = 0   # 当前时间
        self.step = step
    def add_timeline(self, time_point):
        # 添加时间节点
        self.timeline.append(time_point)

    def delete_timeline(self, time_point):
        # 删除时间节点,是否切换会比较好
        self.timeline.remove(time_point)

    def get_next_point(self):
        # 到达该时间节点
        self.current_time = min(self.timeline)
