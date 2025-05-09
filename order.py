import random
from Config import work_name_order


class Order:
    def __init__(self, order_id, all_zones):
        self.order_id = order_id  # 订单ID
        self.zones = self.generate_order_zones(all_zones)  # 订单所需的生产区顺序

    import random

    def generate_order_zones(self, all_zones):
        """
        生成一个订单所需的生产区顺序。
        - 订单总是从生产区 0 开始
        - 后续的生产区可以从剩余的区中随机选择
        - 不一定使用所有的生产区，可以通过 max_zones 控制使用的车间数量
        """
        # 确保从生产区 0 开始，经过生产区 1
        zones = [0, 1]

        # 剩余的生产区（不包含0、1、6号区）[2,3,4,5]
        remaining_zones = list(set(range(2, len(all_zones))) - {6})
        max_zones = random.randint(1,7)

        num_zones_to_select = min(max_zones - 1, len(remaining_zones))  # 除了0、1号区外，最多选择的数量

        # 随机选择剩余的生产区
        random.shuffle(remaining_zones)
        selected_zones = remaining_zones[:num_zones_to_select]

        # 将选择的生产区加入订单的生产区顺序
        zones.extend(selected_zones)

        # 添加最后必须经过的6号区
        zones.append(6)

        # 使用字典转换数字为汉字
        zones = [work_name_order[zone] for zone in zones]  # 假设 work_name_order 是你定义的映射字典
        """[0,1,x,x,x,x,6]"""
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
