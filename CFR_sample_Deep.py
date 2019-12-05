from graph import Graph
import numpy as np
import datetime
import torch
import CFR_outcomesample as cfr
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data

N_iteration = 100 #number of iteration when training
LearningRate_adv = 0.001#learning rate of advantage network
Iteration = 100 #number of iteration in CFR
N_traversal = 10 #number of sampling the game tree
n_player = 3
Horizon = 3
n_police = 2
BATCH_SIZE = 50
#test

#define loss function
def my_loss(x, y):
    loss = 0
    for i in range(len(x)):
        loss += (y[i][0] + 1) * torch.pow((x[i] - y[i][1]), 2)
    return loss/len(x)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x


def train_network(memory, model):#train an advantage network here
    optimizer = torch.optim.SGD(model.parameters(), lr=LearningRate_adv)
    plt_loss = []
    #train_data = torch.empty(11,1, dtype=torch.float64)
    train_data = torch.from_numpy(memory[0][0])
    train_data = train_data.unsqueeze(0)
    target_data = torch.tensor([memory[0][1], memory[0][2]])
    target_data = target_data.unsqueeze(0)
    #print(target_data)
    for i in range(len(memory)):
        if i != 0:
            b = torch.from_numpy(memory[i][0])
            b = b.unsqueeze(0)
            c = torch.tensor([memory[i][1], memory[i][2]])
            c = c.unsqueeze(0)
            train_data = torch.cat((train_data, b), dim=0, out=None)
            target_data = torch.cat((target_data, c), dim=0, out=None)
    torch_dataset = Data.TensorDataset(train_data, target_data)
    loader = Data.DataLoader(dataset = torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    for t in range(N_iteration):
        for step, (batch_x, batch_y) in enumerate(loader):
            out = model(batch_x)
            loss = my_loss(out, batch_y)  # loss是定义为神经网络的输出与样本标签y的差别，故取softmax前的值
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            plt_loss.append(loss.item())
            print('interation:',t,'|step: ',step, '|loss: ', loss.item())
        #if t % 50 == 0:
            #print("t:{}, loss:{}".format(t, loss.item()))
    #t = [i for i in range(N_iteration)]
    #plt.plot(plt_loss)
    #plt.show()
    return model


def cfr_traversal(history, player, model_1, model_2, memory_1, memory_2, t, info_set, graph):
    #print(history)
    if cfr.is_terminal(history):
        return cfr.terminal_util(history, player)
    n = len(history)
    is_player = n % 2
    info = cfr.get_information_set(info_set, history, graph)
    if is_player == player:
        strategy = get_strategy(model_1, info)
        v = np.zeros(info.n_actions)
        utility = 0
        for a in info.action:
            p = info.action.index(a)
            next_history = history[:]
            next_history.append(a)
            v[p] = cfr_traversal(next_history, player, model_1, model_2, memory_1, memory_2, t, info_set, graph)
            #print(next_history, v[p])
            utility += v[p] * strategy[p]
        regret = v - utility
        memory_1 = memory_add(memory_1, info, t, regret)
        return utility
    else:
        strategy = get_strategy(model_2, info)
        memory_2 = memory_add(memory_2, info, t, strategy)
        a, action_p = cfr.sample_action(info, strategy)
        next_history = history[:]
        next_history.append(a)
        return cfr_traversal(next_history, player, model_1, model_2, memory_1, memory_2, t, info_set, graph)


def get_strategy(model, info):
    regret = np.zeros(len(info.action))
    i = 0
    for a in info.action:
        data = connect(info, a)
        data = torch.from_numpy(data)
        regret[i] = model(data)
        i = i + 1
    total = float(sum(regret))
    if total > 0:
        strategy = regret / total
    else:
        strategy = np.zeros(info.n_actions) + 1. / float(info.n_actions)
    return strategy


def connect(info, a):#链接info和a，生成并返回tensor结构的数据
    data = np.zeros((Horizon) * n_player + n_police)
    j = 0
    for i in info.history[0]:
        if isinstance(i, int):
            data[j] = i
            j = j + 1
        else:
            for z in range(len(i)):
                data[j] = i[z]
                j = j + 1
    j = 0
    if isinstance(a, int):
        data[(Horizon) * n_player] = a
    else:
        for z in range(len(a)):
            data[(Horizon) * n_player + j] = a[z]
            j = j + 1
    return data


def memory_add(memory, info, t, regret): #讲后面的参数放入memory中
    p = 0
    for a in info.action:
        data = connect(info, a)
        experience = (data, t, regret[p])
        p = p + 1
        memory.append(experience)
    return memory


def real_play(model_1, model_2, history, graph, info_set):
    if cfr.is_terminal(history):
        return cfr.terminal_util(history, 1)
    else:
        n = len(history)
        is_player = n % 2
        info = cfr.get_information_set(info_set, history, graph)
        if is_player == 0:
            strategy = get_strategy(model_1, info)
            v = np.zeros(info.n_actions)
            utility = 0
            for a in info.action:
                p = info.action.index(a)
                next_history = history[:]
                next_history.append(a)
                v[p] = real_play(model_1, model_2, next_history, graph, info_set)
                utility += v[p] * strategy[p]
            return utility
        else:
            strategy = get_strategy(model_2, info)
            v = np.zeros(info.n_actions)
            utility = 0
            for a in info.action:
                p = info.action.index(a)
                next_history = history[:]
                next_history.append(a)
                v[p] = real_play(model_1, model_2, next_history, graph, info_set)
                utility += v[p] * strategy[p]
            return utility


def main():
    length = 3
    width = 3
    graph = Graph(length, width)
    info_set = {}
    history = [2, (1, 6)]
    adv_model_1 = Net((Horizon) * n_player + n_police, 50, 1)
    adv_model_2 = Net((Horizon) * n_player + n_police, 50, 1)
    str_model_1 = Net((Horizon) * n_player + n_police, 50, 1)
    str_model_2 = Net((Horizon) * n_player + n_police, 50, 1)
    adv_memory_1 = []
    adv_memory_2 = []
    strategy_memory_1 = []
    strategy_memory_2 =[]
    utility = []
    for t in range(Iteration):
       for k in range(N_traversal):
            cfr_traversal(history, 0, adv_model_1, adv_model_2, adv_memory_1, strategy_memory_2, t, info_set, graph)
       for k in range(N_traversal):
            cfr_traversal(history, 1, adv_model_2, adv_model_1, adv_memory_2, strategy_memory_1, t, info_set, graph)
       start_time = datetime.datetime.now()
       adv_model_1= train_network(adv_memory_1, adv_model_1)
       end_time = datetime.datetime.now()
       print("interation:{}, player 1 length of memory:{}, time:{}".format(t, len(adv_memory_1), end_time-start_time))
       start_time = datetime.datetime.now()
       adv_model_2= train_network(adv_memory_2, adv_model_2)
       end_time = datetime.datetime.now()
       print("interation:{}, player 2 length of memory:{}, time:{}".format(t, len(adv_memory_2), end_time-start_time))
       str_model_1 = train_network(strategy_memory_1, str_model_1)
       str_model_2 = train_network(strategy_memory_2, str_model_2)
       utility.append(real_play(str_model_1, str_model_2, history, graph, info_set))
       print("utility: ", utility)
    print(utility)
    plt.plot(utility)
    plt.show()

if __name__ == '__main__':
    main()
    #for name_str, param in model.named_parameters():
    #    print("{:21} {:19} {}".format(name_str, str(param.shape), param))
    history = [2, (0, 1)]
    key = str(history)
    action = [2, 4]
    n_actions = 2
    info = cfr.InformationSet(key, history, action, n_actions)
    print(connect(info, (1,2)))
