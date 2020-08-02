import numpy as np
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms

from cnn_model import CnnModel
from config import Config
from logger import Logger


class Environment:
    default_states = torch.Tensor([512, 3, 5, 2, 2]).cuda()
    default_deltas = [32, 1, 1, 1, 1]
    default_ranges = [1024, 10, 10, 10, 10]
    action_space = [
        'of_filter_sub',
        'of_filter_add',
        'filter_height_sub',
        'filter_height_add',
        'filter_width_sub',
        'filter_width_add',
        'stride_height_sub',
        'stride_height_add',
        'stride_width_sub',
        'stride_width_add',
        'none'
    ]

    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    _train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=_transform)
    _train_loader = torch.utils.data.DataLoader(_train_set, batch_size=128, shuffle=True, num_workers=2)

    _test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=_transform)
    _test_loader = torch.utils.data.DataLoader(_test_set, batch_size=128, shuffle=False, num_workers=2)

    _device = torch.device('cuda')

    def __init__(self, states=None, deltas=None, ranges=None):
        self.states = Environment.default_states if not states else states
        self.deltas = Environment.default_deltas if not deltas else deltas
        self.ranges = Environment.default_ranges if not ranges else ranges
        self.last_accuracy = 0

    def step(self, action):
        # even for subtract, odd for add
        idx = action // 2
        act = action % 2

        Logger.stage('step', f"from state {self.states}")
        Logger.stage('step', f"action selected: idx{idx} act{act} ({Environment.action_space[action]})")

        if not Environment.action_space[action] == 'none':
            delta = self.deltas[idx] * (act * 2 - 1)
            states = self.states.clone().detach()   # new memory and leave computation graph

            states[idx] += delta
            states[idx] = min(max(states[idx], 1), self.ranges[idx])
            if not Environment._is_state_available(states):
                Logger.stage('step', f"new state is not available; not change")
            else:
                self.states[idx] = min(max(self.states[idx]+delta, 1), self.ranges[idx])

        Logger.stage('step', f"to state {self.states}")
        # self.states = self.states.detach()

        return self.states, Environment.get_reward(self.states)

    @staticmethod
    def _is_state_available(states):
        height = ((8 - states[1]) // states[3] + 1) // 2
        width = ((8 -states[2]) // states[4] + 1) // 2
        return states[0] * height * width != 0

    @staticmethod
    def _test_cnn_model(model, log=True):
        correct, total = .0, .0
        with torch.no_grad():
            for images, labels in Environment._test_loader:
                images, labels = images.to(Environment._device), labels.to(Environment._device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        if log:
            Logger.stage('test', f"acc: {accuracy}")
        return accuracy

    @staticmethod
    def get_reward(states):
        cnn = CnnModel.from_states(states).cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), lr=0.001)

        log_interval = (Config.N_CNN_EPOCH // 20 + 1)
        totalLoss = .0

        Logger.stage('train', 'start')
        for epoch in range(Config.N_CNN_EPOCH):

            for data in Environment._train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(Environment._device), labels.to(Environment._device)

                optimizer.zero_grad()
                loss = criterion(cnn(inputs), labels)
                loss.backward()

                optimizer.step()
                totalLoss += loss.item()

            if (epoch + 1) % log_interval == 0:
                acc = Environment._test_cnn_model(cnn, log=False)
                Logger.print(f'[Epoch {epoch}] loss: {totalLoss / len(Environment._train_loader)} acc: {acc}')
                totalLoss = .0

        acc = Environment._test_cnn_model(cnn, log=True)
        reward = acc
        Logger.stage('reward', f'finished {reward}')
        return reward
