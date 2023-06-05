

import random

class SlotMachine:
    def __init__(self, num_levers):
        self.num_levers = num_levers
        self.levers = [0] * num_levers  # 初始化每个拉杆的赔率

    def pull_lever(self, lever_index):
        if lever_index < 0 or lever_index >= self.num_levers:
            raise ValueError("Invalid lever index")
        print(f"level_index={lever_index}")
        # 模拟根据赔率随机决定是否中奖
        if random.random() < self.levers[lever_index]:
            return True  # 中奖
        else:
            return False  # 未中奖

    def set_lever_odds(self, lever_index, odds):
        if lever_index < 0 or lever_index >= self.num_levers:
            raise ValueError("Invalid lever index")
        if odds < 0 or odds > 1:
            raise ValueError("Invalid odds value")

        self.levers[lever_index] = odds

# 示例用法
num_levers = 10
slot_machine = SlotMachine(num_levers)

# 设置每个拉杆的赔率
for i in range(num_levers):
    odds = random.uniform(0, 1)  # 随机生成赔率，范围为 [0, 1]
    slot_machine.set_lever_odds(i, odds)

# 玩家拉动拉杆
lever_to_pull = random.randint(0, num_levers-1)  # 随机选择一个拉杆
result = slot_machine.pull_lever(lever_to_pull)

if result:
    print("恭喜，中奖了！")
else:
    print("很遗憾，未中奖。")
