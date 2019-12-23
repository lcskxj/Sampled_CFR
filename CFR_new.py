from graph import Graph
import numpy as np
import itertools
import csv
import random
import datetime
import matplotlib.pyplot as plt

EPS = 1E-3
Horizon = 5
Delta = 0.8
iteration = 500

class InformationSet(object):
    def __init__(self, key, history, action, n_actions):
        self.key = key
        self.player = len(history) % 2
        self.history = [history]
        self.n_actions = n_actions
        self.action = action
        self.regret_sum = np.zeros(n_actions)
        self.strategy_sum = np.zeros(n_actions) + 1. / float(n_actions)
        self.strategy = np.zeros(n_actions) + 1. / float(n_actions)
        self.reach_pr = 0.
        #self.counter = 0
        if self.player == 0:
            self.mark = history[:-1]
        else:
            self.mark = []
            for j in range(1, len(history), 2):
                self.mark.append(history[j])

    def get_average_strategy(self):
        total = sum(self.strategy_sum)
        if total > 0:
            strategy = self.strategy_sum / float(total)
            strategy = np.where(strategy < EPS, 0., strategy)
            total = sum(strategy)
            strategy /= float(total)
        else:
            strategy = np.zeros(self.n_actions) + 1. / float(self.n_actions)
        return strategy


def match(history1, historylist):  # if the history1 is in the information set
    flag = 0
    temp_history = historylist
    for j in range(1, len(history1), 2):
        if history1[j] == temp_history[j]:
            flag = 1
        else:
            flag = 0
            break
    if flag == 1:
        return True
    else:
        return False


def build_info(info_set, graph, history):  # bulid information sets
    temp = 0
    if len(history) % 2 == 1:
        for dix, info in info_set.items():
            history_temp = info.history[0]
            if len(history) == len(history_temp):
                if match(history, history_temp):
                    info.history.append(history)
                    info1 = info
                    temp = 1
                    break
    else:
        for dix, info in info_set.items():
            history_temp = info.history[0]
            if len(history) == len(history_temp):
                if history[:len(history)-1] == history_temp[:len(history_temp)-1]:
                    info.history.append(history)
                    info1 = info
                    temp = 1
                    break
    if temp == 0:
        key = str(history)
        info1 = None
        action = next_action(graph, history)
        n_action = len(action)
        info1 = InformationSet(key, history, action, n_action)
        info_set[key] = info1
    return info1


def get_information_set(info_set, history, graph):#According to history, return information set I
    temp = 0
    for dix, info in info_set.items():
        history_temp = info.history
        if history in history_temp:
            info1 = info
            temp = 1
    if temp == 0:#if there is no information set containing history, then build one
        info1 = build_info(info_set, graph, history)
    return info1


def next_action(graph, history):
    if (len(history)) % 2 == 0:
        action = history[-2]
        action_next = graph.adj[action - 1]
    else:
        action = history[-2]
        b = [0] * len(action)
        for i in range(len(action)):
            b[i] = graph.adj[action[i] - 1]
        action_next = list(itertools.product(*b))
    return action_next


# 判断是否是terminal node
def is_terminal(history):
    if len(history) < 2 or len(history) % 2 == 1:  # history 长度小于2说明没有完整进行一轮， 除2余数为1说明轮到player2进行action selection，都表明没有结束
        return False
    elif len(history) / 2 >= Horizon:
        return True
    else:
        pursuer_action = history[-1]
        evader_action = history[-2]
        if evader_action in pursuer_action:
            return True
        else:
            return False


# utility of terminal node
def terminal_util(history, player):
    n = len(history) // 2
    pursuer_action = history[-1]
    evader_action = history[-2]
    if evader_action in pursuer_action:
        if player == 0:
            return -10 * Delta ** n
        else:
            return 10 * Delta ** n
    else:
        if player == 0:
            return 10 #* Delta ** n
        else:
            return -10 #* Delta ** n


def cfr(info_set, graph, history, player, pr_1=1., pr_2=1.):
    #print(history, is_terminal(history), player)
    if is_terminal(history):
        utility = terminal_util(history, player)
        #print(utility)
        return utility
    n = len(history)
    info = get_information_set(info_set, history, graph)
    strategy = info.strategy
    is_player = n % 2
    if is_player == player:
        if info.reach_pr == 0.:
            info.reach_pr += pr_1
        #print('3435',info.key, info.reach_pr)
    action_utils = np.zeros(info.n_actions)
    if is_player == player:
        available_action = info.action
        for i, action in enumerate(available_action):
            next_history = history[:]
            next_history.append(action)
            action_utils[i] = cfr(info_set, graph, next_history, player, pr_1 * strategy[i], pr_2)
    else:
        available_action = next_action(graph, history)
        for i, action in enumerate(available_action):
            next_history = history[:]
            next_history.append(action)
            action_utils[i] = cfr(info_set, graph, next_history, player, pr_1, pr_2 * strategy[i])
    util = sum(action_utils * strategy)
    if is_player == player:
        regrets = action_utils - util
        info.regret_sum += pr_2 * regrets
    return util


#regret matching
def update_strategy(info):
    regret = np.where(info.regret_sum > 0, info.regret_sum, 0)
    total = float(sum(regret))
    if total > 0:
        strategy = regret / total
    else:
        strategy = np.zeros(info.n_actions) + 1. / float(info.n_actions)
        #strategy = strategy / sum(strategy)
    return strategy


def test_cfr(info_set, graph, history):
    utility = []
    for i in range(iteration):
        print(i)
        cfr(info_set, graph, history, 0)
        cfr(info_set, graph, history, 1)
        #  更新策略
        for _, info in info_set.items():
            info.strategy_sum += info.reach_pr * info.strategy
            #print(info.mark, info.reach_pr, info.action, info.strategy)
            info.strategy = update_strategy(info) # regret matching
            #print(info.strategy)
            info.reach_pr = 0.
        utility.append(real_play(history, graph, info_set))
    print("Size of info_set: ", len(info_set))
    print(utility)
    plt.plot(utility)
    plt.show()
    # show the results
    '''result_list = []
    count_history = 0
    for dix, info in info_set.items():
        total = sum(info.strategy_sum)
        info.strategy = info.strategy_sum / float(total)
        count_history += len(info.history)
        result_list.append([info.history, info.action, info.strategy])
    print(count_history)
    with open("result4.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result_list)
    print('ok')'''


def real_play(history, graph, info_set):
    if is_terminal(history):
        return terminal_util(history, 1)
    else:
        n = len(history)
        info = get_information_set(info_set, history, graph)
        strategy = info.get_average_strategy()
        v = np.zeros(info.n_actions)
        utility = 0
        for p, a in enumerate(info.action):
            next_history = history[:]
            next_history.append(a)
            v[p] = real_play(next_history, graph, info_set)
            utility += v[p] * strategy[p]
        return utility


def main():
    length = 3
    width = 3
    #  定义game
    #  1.1 图结构
    graph = Graph(length, width)
    history = [2, (1, 6)]
    #  test cfr
    info_set = {}
    start_time = datetime.datetime.now()
    test_cfr(info_set, graph, history)#test CFR
    #test_cfros(info_set, graph, history)
    #  结果
    #print(history)
    #Utility = real_play(history, graph, info_set)
    #print("utility: ", Utility)
    end_time = datetime.datetime.now()
    print(end_time - start_time)


if __name__ == '__main__':
    main()