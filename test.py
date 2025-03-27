import pickle

# 假设这些是您要保存的数据
energy_pic = [1, 2, 3, 4]
time_pic = [4, 3, 2, 1]
agv_distributions = [[1, 0], [2, 1], [1, 1], [0, 0]]
order_distributions = [[1, 2], [2, 3], [3, 4], [4, 5]]
distributions_dicts = [{'zone1': [1, 2]}, {'zone2': [2, 3]}]

# 将数据存储到文件
data = {
    'energy': energy_pic,
    'time': time_pic,
    'agv_distribution': agv_distributions,
    'order': order_distributions,
    'distributions_dict': distributions_dicts
}

with open('data.pkl', 'wb') as file:
    pickle.dump(data, file)
