import random

import numpy as np


class BernoulliBandit:
    """拉杆数为k的多臂老虎机：
    1.拉动每根拉杆的奖励服从伯努利分布（Bernoulli distribution）
    2.每次拉下拉杆有p的概率获得的奖励为1，有1-p的概率获得的奖励为0
    3.奖励为1代表获奖，奖励为0代表没有获奖
    """

    def __init__(self, k):
        self.k = k
        self.probs = np.random.uniform(size=k)
        self.best_idx = np.argmax(self.probs)
        self.best_probs = self.probs[self.best_idx]

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            # print(f"拉动{k}号位赌博机，中奖")
            return 1
        else:
            # print(f"拉动{k}号位赌博机，没有中奖")
            return 0

    def print_prob(self):
        for i in range(self.k):
            res = self.step(i)
            print(f"{i}号拉杆后：获奖={res}，获奖概率为{self.probs[i]}")


if __name__ == '__main__':
    np.random.seed(1)
    arms = 10
    arm = random.randint(0, arms-1)
    bandits_10_arm = BernoulliBandit(arms)
    bandits_10_arm.step(k=arm)
    print(f"随机生成了{arms}臂伯努利老虎机")
    print(f"获奖概率最大的拉杆为{bandits_10_arm.best_idx}，其获奖概率为{bandits_10_arm.best_probs}")
    bandits_10_arm.print_prob()

    # 随机生成了10臂伯努利老虎机
    # 获奖概率最大的拉杆为1，其获奖概率为0.7203244934421581
    # 0号拉杆后：获奖 = 0，获奖概率为0.417022004702574
    # 1号拉杆后：获奖 = 1，获奖概率为0.7203244934421581
    # 2号拉杆后：获奖 = 0，获奖概率为0.00011437481734488664
    # 3号拉杆后：获奖 = 1，获奖概率为0.30233257263183977
    # 4号拉杆后：获奖 = 0，获奖概率为0.14675589081711304
    # 5号拉杆后：获奖 = 0，获奖概率为0.0923385947687978
    # 6号拉杆后：获奖 = 0，获奖概率为0.1862602113776709
    # 7号拉杆后：获奖 = 1，获奖概率为0.34556072704304774
    # 8号拉杆后：获奖 = 1，获奖概率为0.39676747423066994
    # 9号拉杆后：获奖 = 0，获奖概率为0.538816734003357


