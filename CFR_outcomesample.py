from graph import Graph
import numpy as np
import itertools
import csv
import random
import networkx as nx
import matplotlib.pyplot as plt
import datetime

Horizon = 3
Delta = 0.9
epsilon = 1


class InformationSet(object):
    def __init__(self, key, history, action, n_actions):
        self.player = len(history) % 2 + 1
        self.key = key
        self.history = [history]
        self.n_actions = n_actions
        self.action = action
        self.regret_sum = np.zeros(n_actions)
        self.strategy_sum = np.zeros(n_actions) + 1. / float(n_actions)
        self.strategy = np.zeros(n_actions) + 1. / float(n_actions)
        self.reach_pr = 0.


def match(history1, history_list):  # if the history1 is in the information set
    flag = 0
    temp_history = history_list
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
    if len(history) / 2 >= Horizon:
        return True
    elif len(history) < 2 or len(history) % 2 == 1:  # history 长度小于2说明没有完整进行一轮， 除2余数为1说明轮到player2进行action selection，都表明没有结束
        return False
    else:
        pursuer_action = history[-1]
        evader_action = history[-2]
        if evader_action in pursuer_action:
        #if evader_action == pursuer_action:
            return True
        else:
            return False


# utility of terminal node
def terminal_util(history, player):
    n = len(history) / 2
    pursuer_action = history[-1]
    evader_action = history[-2]
    if n <= Horizon:
        if evader_action in pursuer_action:
        #if evader_action == pursuer_action:
            if player == 0:
                return -1 * Delta ** n
            else:
                return Delta ** n
        else:
            if player == 0:
                return 1
            else:
                return -1


def sample_action(info, sample_strategy):
    temp = random.randint(1, 100000) / 100000.
    strategy_sum = 0
    for i in range(0, info.n_actions):
        strategy_sum += sample_strategy[i]
        if temp <= strategy_sum:
            action = info.action[i]
            action_probability = sample_strategy[i]
            break
        elif i == info.n_actions - 1:
            action = info.action[i]
            action_probability = sample_strategy[i]
            break
    return action, action_probability


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
            #print('player1', next_history, pr_1 * strategy[i], action_utils[i])
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


def cfr_out_sample(info_set, graph, history, player, opponent_p=1., sample_p=1.):
    if is_terminal(history):
        utility = terminal_util(history, player) #/ sample_p
        #print('utility', utility, sample_p)
        return utility
    # if chance node
    n = len(history)
    is_player = n % 2 + 1
    if is_player != player:
        info = get_information_set(info_set, history, graph)
        strategy = update_strategy(info)
        for a in info.action:
            p = info.action.index(a)
            info.strategy_sum[p] += strategy[p] / sample_p
        action, action_probability = sample_action(info, strategy)
        #print('  ', action)
        next_history = history[:]
        next_history.append(action)
        return cfr_out_sample(info_set, graph, next_history, player, opponent_p * action_probability, sample_p)
    info = get_information_set(info_set, history, graph)
    strategy = update_strategy(info)
    sample_strategy = epsilon * (np.zeros(info.n_actions) + 1. / float(info.n_actions)) + (1 - epsilon) * strategy
    action, action_probability = sample_action(info, sample_strategy)
    #print(action)
    next_history = history[:]
    next_history.append(action)
    v = np.zeros(info.n_actions)
    utility = 0
    for a in info.action:
        p = info.action.index(a)
        if a == action:
            v[p] = cfr_out_sample(info_set, graph, next_history, player, opponent_p, sample_p * action_probability)
        else:
            v[p] = 0
        utility += v[p] * strategy[p]
        #print(' ', utility)
    for a in info.action:
        p = info.action.index(a)
        regret = v[p] - utility
        info.regret_sum[p] += opponent_p * regret
    #print(history, v, info.action, strategy, utility)
    return utility


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


def test_cfr_out_sample(info_set, graph, history):
    iteration = 10
    result_value1 = []
    result_value2 = []
    for i in range(1, iteration+1):
        result_value1 = cfr_out_sample(info_set, graph, history, 1)
        #print(result_value1)
        result_value2 = cfr_out_sample(info_set, graph, history, 2)
        #print(result_value2)
        for _, info in info_set.items():
            info.strategy_sum += info.reach_pr * info.strategy
            info.strategy = update_strategy(info)
            info.reach_pr = 0.
    count_history = 0
    cfr(info_set, graph, history)
    # show the results
    result_list = []
    #sorted(info_set.keys())
    for dix, info in info_set.items():
        total = sum(info.strategy_sum)
        info.strategy = info.strategy_sum / float(total)
        count_history += len(info.history)
        result_list.append([info.player, info.history, info.action, info.strategy])
    print(len(info_set))
    print(count_history)
    with open("result6.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result_list)
    print('ok')


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
    #test_cfr(info_set, graph, history)#test CFR
    test_cfr_out_sample(info_set, graph, history)
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    #  结果可视化


if __name__ == '__main__':
    main()
    """G = nx.Graph()
    for i in range(0, graph.node):
       G.add_node(i)
    for i in range(0, graph.node):
        for j in range(0, len(graph.adj[i])):
            G.add_edge(i, graph.adj[i][j])
    nx.draw(G,
        with_labels=True, #这个选项让节点有名称
        edge_color='b', # b stands for blue!
        pos=nx.spring_layout(G), # 这个是选项选择点的排列方式，具体可以用 help(nx.drawing.layout) 查看
     # 主要有spring_layout  (default), random_layout, circle_layout, shell_layout
     # 这里是环形排布，还有随机排列等其他方式
        node_color='r', # r = red
        node_size=100, # 节点大小
        width=3, # 边的宽度
       )
    plt.show()"""