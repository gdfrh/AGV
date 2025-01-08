import time

# 定义一个全局时间变量
t = time.time()  # 初始全局时间，单位可以是秒

def update_time():
    global t
    t = time.time()  # 更新全局时间为当前时间戳

def get_current_time():
    global t
    return t

def run_task():
    global t
    update_time()
    print(f"Task started at time {get_current_time()}")
    # 假设任务耗时2秒
    time.sleep(2)
    update_time()
    print(f"Task completed at time {get_current_time()}")
