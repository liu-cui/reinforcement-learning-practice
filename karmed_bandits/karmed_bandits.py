import random

import matplotlib.pyplot as plt
import numpy as np
from typing import List


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


class Solver:
    def __init__(self, bandit: BernoulliBandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.k)
        self.regret = 0
        self.actions = []
        self.regrets = []

    def update_regret(self, k):
        self.regret += self.bandit.best_probs - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    def __init__(self, bandit: BernoulliBandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.k)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.k)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit: BernoulliBandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.k)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.k)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


class UCB(Solver):
    def __init__(self, coef, bandit: BernoulliBandit, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.k)
        self.total_count = 0
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


def plot_results(solvers: List[Solver], solver_names: List[str]):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title(f"{solvers[0].bandit.k}-armed bandit")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    arms = 10
    arm = random.randint(0, arms - 1)
    bandit_10_arm = BernoulliBandit(arms)
    bandit_10_arm.step(k=arm)
    print(f"随机生成了{arms}臂伯努利老虎机")
    print(f"获奖概率最大的拉杆为{bandit_10_arm.best_idx}，其获奖概率为{bandit_10_arm.best_probs}")
    # bandits_10_arm.print_prob()
    np.random.seed(1)
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print(f"epsilon={epsilon_greedy_solver.epsilon}的贪婪算法的累积懊悔为：{epsilon_greedy_solver.regret}")
    # plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])
    np.random.seed(0)
    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons]
    epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    for epsilon_greedy_solver in epsilon_greedy_solver_list:
        epsilon_greedy_solver.run(5000)
        print(f"epsilon={epsilon_greedy_solver.epsilon}的贪婪算法的累积懊悔为：{epsilon_greedy_solver.regret}")
    # plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)
    np.random.seed(1)
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
    decaying_epsilon_greedy_solver.run(5000)
    print(f"epsilon值衰减的贪婪算法的累积懊悔为：{decaying_epsilon_greedy_solver.regret}")
    # plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

    np.random.seed(1)
    coef = 1  # 控制不确定性比重的系数
    UCB_solver = UCB(coef, bandit_10_arm)
    UCB_solver.run(5000)
    print(f"UCB上置信界算法的累积懊悔为：{UCB_solver.regret}")
    # plot_results([UCB_solver], ["UCB"])


