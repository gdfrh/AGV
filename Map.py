import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class MapSpace:
    def __init__(self, rows, cols, fill_char, road_width,work_width_up,work_width_down, work_name_up, work_name_down):
        # 使用 NumPy 创建一个二维数组，填充默认字符
        self.rows = rows
        self.cols = cols
        self.road_width = road_width  # 设置道路的行数
        self.space = np.full((rows, cols), fill_char, dtype=str)
        self.work_width_up = work_width_up
        self.work_width_down = work_width_down
        self.work_name_up= work_name_up
        self.work_name_down = work_name_down
        # 计算中间道路的起始行和结束行
        start_row = (self.space.shape[0] // 2) - (self.road_width // 2)
        end_row = start_row + self.road_width - 1
        self.start_row = start_row
        self.end_row = end_row
        # 初始化车间布局
        self._init_workshop_layout()
        self.temp=0

    def _init_workshop_layout(self):
        """初始化车间的布局，包括外墙、内墙、出入口和道路"""
        # 设置车间外墙
        for i in range(self.space.shape[0]):
            self.space[i, 0] = '#'
            self.space[i, -1] = '#'
        for j in range(self.space.shape[1]):
            self.space[0, j] = '#'
            self.space[-1, j] = '#'

        # 设置车间内部墙壁
        for i in range(1, self.start_row):
            for j in self.work_width_up:
                self.space[i, j] = '#'

        for i in range(self.end_row, self.rows):
            for j in self.work_width_down:
                self.space[i, j] = '#'

        # 设置中间n行为道路
        self.set_road()

    def set_road(self):
        """设置地图中间n行为道路"""
        # 设置中间n行为道路
        for row in range(self.start_row,self.end_row + 1):
            for col in range(self.space.shape[1]):
                self.space[row, col] = 'R'

    def set_value(self, x, y, value):
        """设置指定位置的字符值"""
        self.space[x, y] = value

    def get_value(self, x, y):
        """获取指定位置的字符值"""
        return self.space[x, y]

    def set_entrance(self, x, y):
        """设置出入口"""
        self.space[x, y] = 'E'

    def display(self):
        """使用 Matplotlib 绘制图形地图"""
        # 定义颜色映射，使用 RGB 值
        color_map = {
            '#': (0.8, 0.8, 0.8),  # 墙壁为灰色 (RGB)
            'R': (0.6, 0.8, 1.0),  # 道路为淡蓝色 (RGB)
            ' ': (1.0, 0.98, 0.9)  # 空白区域为米色 (RGB)
        }

        # 将字符数组转换为数值数组，数值对应颜色映射
        color_indices = np.vectorize(lambda x: {'#': 0, 'R': 1, ' ': 2}[x])(self.space)

        # 使用 RGB 的颜色列表
        colors = [(0.8, 0.8, 0.8), (0.6, 0.8, 1.0), (1.0, 0.98, 0.9)]  # 对应的 RGB 颜色

        # 创建一个自定义的色图
        cmap = mcolors.ListedColormap(colors)
        # 创建一个绘图区域
        fig, ax = plt.subplots()

        # 绘制图像
        im = ax.imshow(color_indices, cmap=cmap, aspect='auto')

        # 设置字体（支持中文）
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体支持中文
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号问题

        # 在地图上添加文字
        j=0
        for i in self.work_width_up:
            ax.text((self.temp+i)/2, (0 + self.start_row) / 2,
                    self.work_name_up[j], ha='center', va='center', color='black',
                    fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
            self.temp=i
            j+=1

        j=0
        self.temp=0
        for i in self.work_width_down:
            ax.text((self.temp+i)/2, (self.end_row + self.rows-1) / 2,
                    self.work_name_down[j], ha='center', va='center', color='black',
                    fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
            self.temp=i
            j+=1

        # 绘制图像
        plt.imshow(color_indices, cmap=cmap, aspect='auto')

        # 隐藏坐标轴
        plt.axis('off')

        # 显示地图
        plt.show()


