"""
我感觉来规划小车的分配需要用到订单，是否可以定义几种订单A B C，
A：1-2-（345）-6-7
B：1-3-（46）-2
C:1-7
像这种的订单随机的来几十个

设定目标函数为时间t来判断小车的分配呢，这样小车的分配就与订单进行绑定了
是否需要考虑路由时间？如何考虑？
"""
import random
from Config import work_name_order



class Order:
    def __init__(self, order_id, all_zones):
        self.order_id = order_id  # 订单ID
        self.zones = self.generate_order_zones(all_zones)  # 订单所需的生产区顺序

    def generate_order_zones(self, all_zones):
        """
        生成一个订单所需的生产区顺序。
        - 订单总是从生产区 0 开始
        - 后续的生产区可以从剩余的区中随机选择
        """
        # 确保从生产区 0 开始
        zones = [0]

        # 剩余的生产区（不包含0号区）
        remaining_zones = list(range(1, len(all_zones)))

        # 随机选择后续的生产区顺序
        random.shuffle(remaining_zones)

        # 将剩余的生产区加入订单的生产区顺序
        zones.extend(remaining_zones)

        zones = [work_name_order[zone] for zone in zones]  # 使用字典转换数字为汉字

        return zones

    def get_order_details(self):
        return {
            "order_id": self.order_id,
            "zones": self.zones,
        }


class OrderManager:
    def __init__(self, all_zones, num_orders):
        self.all_zones = all_zones  # 所有的生产区
        self.orders = self.generate_orders(num_orders)  # 生成订单

    def generate_orders(self, num_orders):
        orders = []
        for i in range(num_orders):
            order = Order(i, self.all_zones)
            orders.append(order)
        return orders

    def print_orders(self):
        for order in self.orders:
            details = order.get_order_details()
            print(f"Order ID: {details['order_id']}, Production Zones: {details['zones']}")


    def get_orders(self):
        orders = []
        for order in self.orders:
            details = order.get_order_details()
            orders.append(details['zones'])


        return orders

