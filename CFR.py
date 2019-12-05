# Author: Kyle Kastner

# License: BSD 3-Clause

# Drectly lifted from the great blogpost by Justin Sermeno https://justinsermeno.com/posts/cfr/

# References:
# https://int8.io/counterfactual-regret-minimization-for-poker-ai/
# http://cs.gettysburg.edu/~tneller/modelai/2013/cfr/cfr.pdf
# https://github.com/Limegrass/Counterfactual-Regret-Minimization/blob/notes/Learning_to_Play_Poker_using_CounterFactual_Regret_Minimization.pdf
# http://poker.cs.ualberta.ca/publications/Burch_Neil_E_201712_PhD.pdf
# http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf
# https://arxiv.org/pdf/1407.5042.pdf
# https://arxiv.org/pdf/1809.03057.pdf
# https://arxiv.org/abs/1811.00164

import numpy as np

N_ACTIONS = 2  # call or bet
N_CARDS = 3  # 3 possible cards
N_ITERATIONS = 1  # number of iterations for CFR
EPS = 1E-3  # value below which to remove option


class InformationSet(object):
    def __init__(self, key):
        self.key = key
        self.regret_sum = np.zeros(N_ACTIONS)
        self.strategy_sum = np.zeros(N_ACTIONS)
        self.strategy = np.zeros(N_ACTIONS) + 1. / float(N_ACTIONS)
        self.reach_pr = 0.

    def next_strategy(self):
        self.strategy_sum += self.reach_pr * self.strategy
        self.strategy = self.calc_strategy(self.reach_pr)
        self.reach_pr = 0.

    def calc_strategy(self, pr):
        strategy = np.where(self.regret_sum > 0, self.regret_sum, 0)
        total = float(sum(strategy))
        if total > 0:
            strategy = strategy / total
        else:
            strategy = np.zeros(N_ACTIONS) + 1. / float(N_ACTIONS)
        return strategy

    def get_average_strategy(self):
        total = sum(self.strategy_sum)
        if total > 0:
            strategy = self.strategy_sum / float(total)
            # remove low prob options
            # called "purification"
            # https://www.cs.cmu.edu/~sandholm/StrategyPurification_AAMAS2012_camera_ready_2.pdf
            strategy = np.where(strategy < EPS, 0., strategy)
            total = sum(strategy)
            strategy /= float(total)
        else:
            strategy = np.zeros(N_ACTIONS) + 1. / float(N_ACTIONS)
        return strategy

    def __repr__(self):
        strategies = ['{:03.2f}'.format(x)
                      for x in self.get_average_strategy()]
        return '{} {}'.format(self.key.ljust(6), strategies)


def is_chance_node(history):
    # only chance node is at the start of the game for Kuhn poker
    return history == ""


def chance_util(i_map):
    expected_value = 0.
    n_possibilities = 6 # 3 choose 1, 2 choose 1 = 6
    for i in range(N_CARDS):
        for j in range(N_CARDS):
            # 3 cards to choose from, then 2
            if i != j:
                expected_value += cfr(i_map, "rr", i, j, 1., 1., 1. / float(n_possibilities))
    return expected_value / float(n_possibilities)


def is_terminal(history):
    # returns True if history is an end state
    possibilities = {"rrcc": True,
                     "rrcbc": True,
                     "rrcbb": True,
                     "rrbc": True,
                     "rrbb": True}
    return history in possibilities


def terminal_util(history, card_1, card_2):
    n = len(history)
    card_player = card_1 if n % 2 == 0 else card_2
    card_opponent = card_2 if n % 2 == 0 else card_1
    if history == "rrcbc" or history == "rrbc":
        # last player folded, current player wins
        return 1.
    elif history == "rrcc":
        # showdown, no bets
        return 1. if card_player > card_opponent else -1.
    # showdown with 1 bet
    assert(history == "rrcbb" or history == "rrbb")
    return 2. if card_player > card_opponent else -2.


def card_str(card):
    # print the name of the card
    if card == 0:
        return "J"
    elif card == 1:
        return "Q"
    return "K"


def get_info_set(i_map, card, history):
    key = card_str(card) + "_" + history
    info_set = None
    if key not in i_map:
        info_set = InformationSet(key)
        i_map[key] = info_set
        #print(i_map)
        return info_set
    return i_map[key]


def cfr(i_map, history="", card_1=-1, card_2=-1, pr_1=1., pr_2=1., pr_c=1.):
    """
    Counterfactual regret minimization algorithm.
    Parameters
    ----------
    i_map: dict
        Dictionary of all information sets.
    history : [{'r', 'c', 'b'}], str
        A string representation of the game tree path we have taken.
        Each character of the string represents a single action:
        'r': random chance action
        'c': check action
        'b': bet action
    card_1 : (0, 2), int
        player A's card
    card_2 : (0, 2), int
        player B's card
    pr_1 : (0, 1.0), float
        The probability that player A reaches `history`.
        1.0 means player A didn't contribute
    pr_2 : (0, 1.0), float
        The probability that player B reaches `history`.
        1.0 means player B didn't contribute
    pr_c: (0, 1.0), float
        The probability contribution of chance events to reach `history`.
        1.0 means chance didn't contribute
    """
    if is_chance_node(history):
        return chance_util(i_map)
    if is_terminal(history):
        return terminal_util(history, card_1, card_2)
    n = len(history)
    is_player_1 = n % 2 == 0  #distinguish which player should play
    info_set = get_info_set(i_map, card_1 if is_player_1 else card_2, history)
    strategy = info_set.strategy
    if is_player_1:
        info_set.reach_pr += pr_1
    else:
        info_set.reach_pr += pr_2
    # counterfactual utility
    action_utils = np.zeros(N_ACTIONS)
    # check or bet
        # DFS recursion
    if is_player_1:
        for i, action in enumerate(["c", "b"]):
            next_history = history + action
            action_utils[i] = -1 * cfr(i_map, next_history,
                                       card_1, card_2,
                                       pr_1 * strategy[i], pr_2, pr_c)
            print('player1', next_history, pr_1 * strategy[i], action_utils[i])
    else:
        for i, action in enumerate(["c", "b"]):
            next_history = history + action
            action_utils[i] = -1 * cfr(i_map, next_history,
                                       card_1, card_2,
                                       pr_1, pr_2 * strategy[i], pr_c)
            print('player2', next_history, pr_2 * strategy[i], action_utils[i])
    # Utility of the information set
    util = sum(action_utils * strategy)
    regrets = action_utils - util
    # CFR+, toward RBP using 4.2 of https://www.cs.cmu.edu/~sandholm/regret-basedPruning.nips15.withAppendix.pdf
    if is_player_1:
        # modified RBP CFR+
        instant_regret = pr_2 * pr_c * regrets
        idx = (info_set.regret_sum <= 0) & (instant_regret > 0)
        info_set.regret_sum[idx] = instant_regret[idx]
        info_set.regret_sum[~idx] += instant_regret[~idx]
        # CFR+
        #info_set.regret_sum = np.maximum(info_set.regret_sum + pr_2 * pr_c * regrets, 0)
        # CFR
        #info_set.regret_sum += pr_2 * pr_c * regrets
    else:
        # modified RBP CFR+
        instant_regret = pr_1 * pr_c * regrets
        idx = (info_set.regret_sum <= 0) & (instant_regret > 0)
        info_set.regret_sum[idx] = instant_regret[idx]
        info_set.regret_sum[~idx] += instant_regret[~idx]
        # CFR+
        #info_set.regret_sum = np.maximum(info_set.regret_sum + pr_1 * pr_c * regrets, 0)
        # CFR
        #info_set.regret_sum += pr_1 * pr_c * regrets
    return util


def display_results(ev, i_map):
    print('player 1 expected value: {}'.format(ev))
    print('player 2 expected value: {}'.format(-1 * ev))
    print('player 1 strategies:')
    sorted_items = sorted(i_map.items(), key=lambda x: x[0])
    for _, v in [s for s in sorted_items if len(s[0]) % 2 == 0]:
        print(v)
    print('player 2 strategies:')
    for _, v in [s for s in sorted_items if len(s[0]) % 2 == 1]:
        print(v)


def main():
    i_map = {} # information sets aka decision nodes
    expected_game_value = 0.
    for _ in range(N_ITERATIONS):
        expected_game_value += cfr(i_map)
        for _, v in i_map.items():
            v.next_strategy()

    expected_game_value /= float(N_ITERATIONS)
   # print(" value : %s " % i_map.items())
    display_results(expected_game_value, i_map)


if __name__ == "__main__":
    main()
