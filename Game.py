

from graph import Graph
import numpy as np
import itertools

EPS = 1E-3
Horizon = 3
Delta = 1


class InformationSet(object):
    def __init__(self, key, n_actions):
        self.key = key
        self.n_actions = n_actions
        self.regret_sum = np.zeros(n_actions)
        self.strategy_sum = np.zeros(n_actions)
        self.strategy = np.zeros(n_actions) + 1. / float(n_actions)
        self.reach_pr = 0.


def next_action(graph, history):
    if (len(history)) % 2 == 0:
        action = history[-2]
        action_next = graph.adj[action]
    else:
        action = history[-2]
        b = [0] * len(action)
        for i in range(len(action)):
            b[i] = graph.adj[action[i]]
        action_next = list(itertools.product(*b))
    return action_next


# 判断是否是terminal node
def is_terminal(history):
    if len(history) >= Horizon * 2:
        return True
    elif len(history) < 2 or len(history) % 2 == 1:  # history 长度小于2说明没有完整进行一轮， 除2余数为1说明轮到player2进行action selection，都表明没有结束
        return False
    else:
        pursuer_action = history[-1]
        evader_action = history[-2]
        if evader_action in pursuer_action:
            return True
        else:
            return False


# utility of terminal node
def terminal_util(history):
    n = len(history) / 2
    pursuer_action = history[-1]
    evader_action = history[-2]
    if n <= Horizon:
        if evader_action in pursuer_action:
            return Delta ** n
        else:
            return 0


def build_info(info_set, graph, history):
    key = str(history)
    info = None
    n_action = len(next_action(graph, history))
    if key not in info_set:
        info = InformationSet(key, n_action)
        info_set[key] = info
    return info_set[key]


def cfr(info_set, graph, history, pr_1=1., pr_2=1.):
    if is_terminal(history):
        utility = terminal_util(history)
        return utility
    n = len(history)
    is_player_1 = n % 2 == 0
    info = build_info(info_set, graph, history)
    strategy = info.strategy
    if is_player_1:
        info.reach_pr += pr_1
    else:
        info.reach_pr += pr_2
    action_utils = np.zeros(info.n_actions)
    # last_history = history
    if is_player_1:
        available_action = next_action(graph, history)
        for i, action in enumerate(available_action):
            next_history = history[:]
            next_history.append(action)
            action_utils[i] = cfr(info_set, graph, next_history, pr_1 * strategy[i], pr_2)
            #print('player1', next_history, pr_1 * strategy[i], action_utils[i])
    else:
        available_action = next_action(graph, history)
        for i, action in enumerate(available_action):
            next_history = history[:]
            next_history.append(action)
            action_utils[i] = cfr(info_set, graph, next_history, pr_1, pr_2 * strategy[i])
            #print('player2', next_history, pr_2 * strategy[i], action_utils[i])
    util = sum(action_utils * strategy)
    regrets = action_utils - util
    if is_player_1:
        info.regret_sum += pr_2 * regrets
    else:
        info.regret_sum += pr_1 * regrets
    return util

#def cfros(info_set, graph, history, pr_1=1., pr_2=1.):



def update_strategy(info):
    regret = np.where(info.regret_sum > 0, info.regret_sum, 0)
    total = float(sum(regret))
    if total > 0:
        strategy = regret / total
    else:
        strategy = np.zeros(info.n_actions) + 1. / float(info.n_actions)
    return strategy


def main():
    length = 2
    width = 2
    iteration = 100
    #  定义game
    #  1.1 图结构
    graph = Graph(length, width)
    history = [2, [0, 1]]
    #  迭代cfr
    info_set = {}
    expected_value = 0
    for _ in range(iteration):
        expected_value += cfr(info_set,  graph, history)
        #  更新策略
        for _, info in info_set.items():
            info.strategy_sum += info.reach_pr * info.strategy
            info.strategy = update_strategy(info)
            info.reach_pr = 0.
    print(len(info_set))
    for dix, info in info_set.items():
        total = sum(info.strategy_sum)
        info.strategy = info.strategy_sum / float(total)
        print(dix, info.strategy)
    #  结果


if __name__ == '__main__':
    main()
    length = 2
    width = 2
    iteration = 1
    #  定义game
    #  1.1 图结构
    graph = Graph(length, width)
    history = [2, [0, 1]]

   # print(next_action(graph, history))
