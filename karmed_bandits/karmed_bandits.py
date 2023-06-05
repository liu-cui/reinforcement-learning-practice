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
            return 1
        else:
            return 0


if __name__ == '__main__':
    np.random.seed(1)
    arms = 10
    bandits_10_arm = BernoulliBandit(arms)
    print(f"随机生成了{arms}臂伯努利老虎机")
    print(f"获奖概率最大的拉杆为{bandits_10_arm.best_idx}，其获奖概率为{bandits_10_arm.best_probs}")
