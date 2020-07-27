import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import numpy as np
import pickle as pkl

from logger import Logger
from cnn_model import CnnModel
from controller_model import ControllerModel
from config import Config
from q_tables import QTables

# Q表
q_tables = QTables()

# ControllerModel
controller = ControllerModel()
controller_criterion = nn.CrossEntropyLoss()
controller_optimizer = optim.Adam(controller.parameters(), lr=0.001)

losses = []

# global variable
device = torch.device('cuda')
train_loader, test_loader = None, None


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    global train_loader, test_loader
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

    iterate()

    Logger.show_params()


def iterate():
    i = 0
    while Config.last_accuracy <= 0.80:
        Logger.iteration_start(i)
        # reward = get_reward(device, train_loader, test_loader)

        obs, nobs, actions, rewards, a_ts = [], [], [], [], []

        q_tables.enable_learn = False
        Config.backup()
        for param_id in range(5):
            Logger.iteration_start(i, f"param #{param_id}")
            # 选择一个参数进行更新，获取前后状态和对应动作
            # 将奖励值送进Q表内强化学习更新五个参数值
            observation, action, observation_ = q_tables.step(param_id)

            Logger.show_params()

            # 训练&获取&测试模型
            model = train_cnn()
            accuracy = test_cnn(model)
            reward = accuracy - Config.last_accuracy

            # 检查正确率，更新正确率最值
            Config.update_acc(accuracy)
            Config.last_accuracy = accuracy

            # q_tables.learn(reward, observation, action, observation_)
            Config.last_a_t = Config.a_t

            # 维护Q表和Controller学习的输入
            obs.append(observation)
            nobs.append(observation_)
            actions.append(action)
            rewards.append(reward)
            a_ts.append(Config.a_t)

            # 恢复状态
            Config.rollback()

            Logger.stage('reward', f'reward: {reward}')

        q_tables.enable_learn = True

        learn(obs, nobs, actions, rewards, a_ts)
        # 选择一个参数进行更新，获取前后状态和对应动作
        # 将奖励值送进Q表内强化学习更新五个参数值
        best_param_id = np.argmax(rewards)
        observation, action, observation_ = q_tables.step(best_param_id)

        model = train_cnn()
        accuracy = test_cnn(model)

        # 检查正确率，更新正确率最值
        Config.update_acc(accuracy)
        Config.last_accuracy = accuracy

        Config.last_a_t = Config.a_t

        i += 1


def learn(obs, nobs, actions, rewards, a_ts):
    for param_id, (ob, nob, action, reward) in enumerate(zip(obs, nobs, actions, rewards)):
        q_tables.learn(reward, ob, action, nob, param_id)

    # TODO: consider rewards are all negative
    best_param_id = np.argmax(rewards)
    a_t_label = torch.autograd.Variable(
        torch.LongTensor([best_param_id]).to(device))  # torch.FloatTensor([np.eye(5)[np.argmax(rewards)]])

    sum_loss = 0
    for a_t in a_ts:
        # a_t = torch.autograd.Variable(a_t, requires_grad=True)

        controller_optimizer.zero_grad()
        loss = controller_criterion(a_t, a_t_label)
        loss.backward(retain_graph=True)
        # Logger.print("W_h", controller.W_h.weight.grad)
        # Logger.print("W_v", controller.W_v.weight.grad)
        # Logger.print("W_g", controller.W_g.weight.grad)
        # Logger.print(a_t.grad)

        controller_optimizer.step()
        sum_loss += loss.item()

    q_tables.step(best_param_id)

    # Logger.print("w_h", controller.W_h.weight)
    # Logger.print("w_g", controller.W_g.weight)
    # Logger.print("w_v", controller.W_v.weight)
    Logger.stage('learn', f"controller loss: f{loss.item() / len(a_ts)}")


def train_cnn():
    # 根据刚才更新的参数，初始化模型
    cnn = CnnModel.from_config(3).to(device)

    # 此处仅以 Config 中的 cnn 模型参数和 h_t, c_t 为输入
    # a_t, h_t, c_t 作为输出: 其中 h_t, c_t 作为 lstm 的结果，a_t 作为注意力的结果
    controller.forward()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    log_interval = 60
    losses.append([])

    Logger.stage('train', 'start')
    for epoch in range(Config.N_RANGE):
        totalLoss = .0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            # print(cnn(inputs).size(), labels.size())
            loss = criterion(cnn(inputs), labels)
            loss.backward()

            optimizer.step()
            totalLoss += loss.item()
            losses[-1].append(loss.item())

            if (i + 1) % log_interval == 0:
                acc = test_cnn(cnn, log=False)
                Logger.print(f'[Epoch {epoch}, Batch {i}] loss: {totalLoss / log_interval} acc: {acc}')
                totalLoss = .0
                pkl.dump(losses, open('pkl/losses.pkl', 'wb'))

    Logger.stage('train', 'finish')
    torch.save(cnn, 'pkl/model.pkl')
    return cnn


def test_cnn(model, log=True):
    correct, total = .0, .0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    if log:
        Logger.stage('test', f"acc: {accuracy}")
    return accuracy


if __name__ == "__main__":
    main()
