from graph import Graph
import numpy as np
import itertools
import csv
import random
import datetime

EPS = 1E-3
Horizon = 5
Delta = 0.9
epsilon = 0.1

class InformationSet(object):
    def __init__(self, key, history, action, n_actions):
        self.key = key
        self.history = [history]
        self.n_actions = n_actions
        self.action = action
        self.regret_sum = np.zeros(n_actions)
        self.strategy_sum = np.zeros(n_actions) + 1. / float(n_actions)
        self.strategy = np.zeros(n_actions) + 1. / float(n_actions)
        self.reach_pr = 0.
        self.counter = 0


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
    if len(history) / 2 - 1 >= Horizon:
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
def terminal_util(history, player):
    n = len(history) / 2 - 1
    pursuer_action = history[-1]
    evader_action = history[-2]
    if n <= Horizon:
        if evader_action in pursuer_action:
            if player == 1:
                return -1 * Delta ** n
            else:
                return Delta ** n
        else:
            if player == 1:
                return 1
            else:
                return -1


def cfr(info_set, graph, history, pr_1=1., pr_2=1.):
    if is_terminal(history):
        utility = terminal_util(history, 1)
        return utility
    n = len(history)
    is_player_1 = n % 2 == 0
    info = get_information_set(info_set, history, graph)
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
            action_utils[i] = -1 * cfr(info_set, graph, next_history, pr_1 * strategy[i], pr_2)
            print('player1', next_history, pr_1 * strategy[i], action_utils[i])
    else:
        available_action = next_action(graph, history)
        for i, action in enumerate(available_action):
            next_history = history[:]
            next_history.append(action)
            action_utils[i] = -1 * cfr(info_set, graph, next_history, pr_1, pr_2 * strategy[i])
            #print('player2', next_history, pr_2 * strategy[i], action_utils[i])
    util = sum(action_utils * strategy)
    regrets = action_utils - util
    if is_player_1:
        info.regret_sum += pr_2 * regrets
    else:
        info.regret_sum += pr_1 * regrets
    return util


def cfros1(info_set, graph, history, time, player, pr_1=1., pr_2=1., sample_p=1.):
    if is_terminal(history):
        utility = terminal_util(history, player)/sample_p
        return utility, 1
    # if chance node
    n = len(history)
    is_player = n % 2 + 1
    info = get_information_set(info_set, history, graph)
    if is_player == 1:
        info.reach_pr += pr_1
    else:
        info.reach_pr += pr_2
    strategy = update_strategy(info)
    if is_player == player:
        sample_strategy = epsilon * (np.zeros(info.n_actions) + 1. / float(info.n_actions)) + (1 - epsilon) * strategy
        total_strategy = float(sum(sample_strategy))
        sample_strategy = sample_strategy / total_strategy
    else:
        sample_strategy = strategy
    temp = random.randint(1, 100) / 100.
    strategy_sum = 0
    for i in range(0, info.n_actions):
        strategy_sum += strategy[i]
        if temp <= strategy_sum:
            action = info.action[i]
            sample_probability = sample_strategy[i]
            action_probability = strategy[i]
            break
        elif i == info.n_actions - 1:
            action = info.action[i]
            sample_probability = sample_strategy[i]
            action_probability = strategy[i]
            break
    next_history = history[:]
    next_history.append(action)
    if is_player == player:
        result = cfros1(info_set, graph, next_history, time, player, pr_1 * action_probability, pr_2, sample_p * sample_probability)
        w = result[0] * pr_2 * result[1]
        for a in info.action:
            p = info.action.index(a)
            if a == action:
                regret = w * (1 - action_probability)
            else:
                regret = -1 * w
            info.regret_sum[p] += regret
    else:
        result = cfros1(info_set, graph, next_history, time, player, pr_1, pr_2 * action_probability, sample_p * sample_probability)
        info.strategy_sum += (time - info.counter) * pr_2 * strategy
        info.counter = time
    return result[0], result[1] * action_probability


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
    iteration = 1
    expected_value = 0
    for _ in range(iteration):
        print(iteration)
        expected_value += cfr(info_set, graph, history)
        print(iteration)
        #  更新策略
        for _, info in info_set.items():
            info.strategy_sum += info.reach_pr * info.strategy
            info.strategy = update_strategy(info) # regret matching
            info.reach_pr = 0.
    print(len(info_set))
    # show the results
    result_list = []
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
    print('ok')


def test_cfros(info_set, graph, history):
    iteration = 1
    result_value1 = []
    result_value2 = []
    for i in range(1, iteration+1):
        result_value1 += cfros1(info_set, graph, history, i, 1)
        result_value2 += cfros1(info_set, graph, history, i, 2)
        #  更新策略
        #print(result_value1, result_value2)
        for _, info in info_set.items():
            info.strategy_sum += info.reach_pr * info.strategy
            info.strategy = update_strategy(info)  # regret matching
            info.reach_pr = 0.
            #print(i, info.history, info.action, info.regret_sum, info.strategy)
    count_history = 0
    # show the results
    result_list = []
    for dix, info in info_set.items():
        total = sum(info.strategy_sum)
        info.strategy = info.strategy_sum / float(total)
        count_history += len(info.history)
        result_list.append([info.history, info.action, info.strategy])
    print(len(info_set))
    print(count_history)
    with open("result5.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result_list)
    #print('ok')

def main():
    length = 3
    width = 3
    #  定义game
    #  1.1 图结构
    graph = Graph(length, width)
    history = [2, (0, 1)]
    #  test cfr
    info_set = {}
    start_time = datetime.datetime.now()
    test_cfr(info_set, graph, history)#test CFR
    #test_cfros(info_set, graph, history)
    #  结果
    end_time = datetime.datetime.now()
    print(end_time - start_time)


if __name__ == '__main__':
    main()