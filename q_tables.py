import torch
from config import Config
from q_table import QLearningTable
from logger import Logger
from q_status import QStatus


class QTables:
    def __init__(self):
        self.enable_learn = True

        # 初始化每个参数的状态空间
        self.RL_of_filter = QStatus(10, 20, 0, 0)
        self.RL_filter_height = QStatus(3, 1, 15, 1)
        self.RL_filter_width = QStatus(5, 1, 25, 1)
        self.RL_stride_height = QStatus(2, 1, 0, 0)
        self.RL_stride_width = QStatus(1, 1, 0, 0)

        # 对每个状态空间初始化Q表
        self.RL_table_of_filter = QLearningTable(actions=list(range(self.RL_of_filter.n_actions)))
        self.RL_table_filter_height = QLearningTable(actions=list(range(self.RL_filter_height.n_actions)))
        self.RL_table_filter_width = QLearningTable(actions=list(range(self.RL_filter_width.n_actions)))
        self.RL_table_stride_height = QLearningTable(actions=list(range(self.RL_stride_height.n_actions)))
        self.RL_table_stride_width = QLearningTable(actions=list(range(self.RL_stride_width.n_actions)))

    def step(self, param_id=None):
        a_t = Config.a_t

        # 判断是哪个参数需要进行强化学习
        action = 0
        if param_id is None:
            param_id_tensor = torch.max(a_t, 1)[1]
            param_id = param_id_tensor.item()

        if param_id == 0:  # ofFilter
            observation = Config.of_filter
            # 做一个延迟等待操作
            self.RL_of_filter.render()
            # 在Q表中使用当前状态值去获取将要发生的动作
            action = self.RL_table_of_filter.choose_action(str(observation))
            # 使用动作去获取更新后的值
            observation_ = self.RL_of_filter.step(action, self.enable_learn)
            # 将其存在Config中等待下次调用
            Config.of_filter = observation_

        elif param_id == 1:  # filterHeight
            observation = Config.filter_height
            self.RL_filter_height.render()
            action = self.RL_table_filter_height.choose_action(str(observation))
            observation_ = self.RL_filter_height.step(action, self.enable_learn)
            Config.filter_height = observation_

        elif param_id == 2:  # filterWidth
            observation = Config.filter_width
            self.RL_filter_width.render()
            action = self.RL_table_filter_width.choose_action(str(observation))
            observation_ = self.RL_filter_width.step(action, self.enable_learn)
            Config.filter_width = observation_

        elif param_id == 3:  # strideHeight
            observation = Config.stride_height
            self.RL_stride_height.render()
            action = self.RL_table_stride_height.choose_action(str(observation))
            observation_ = self.RL_stride_height.step(action, self.enable_learn)
            Config.stride_height = observation_

        else:  # strideWidth
            observation = Config.stride_width
            self.RL_stride_width.render()
            action = self.RL_table_stride_width.choose_action(str(observation))
            observation_ = self.RL_stride_width.step(action, self.enable_learn)
            Config.stride_width = observation_

        Logger.print("parameter selected: ", param_id)
        Logger.print("action: ", action)

        return observation, action, observation_

    def learn(self, reward, observation, action, observation_, param_id=None):
        a_t = Config.last_a_t

        # 判断是哪个参数需要进行强化学习
        if param_id is None:
            param_id_tensor = torch.max(a_t, 1)[1]
            param_id = param_id_tensor.item()

        if param_id == 0:  # ofFilter
            self.RL_table_of_filter.learn(str(observation), action, reward, str(observation_))
        elif param_id == 1:  # filterHeight
            self.RL_table_filter_height.learn(str(observation), action, reward, str(observation_))
        elif param_id == 2:  # filterWidth
            self.RL_table_filter_width.learn(str(observation), action, reward, str(observation_))
        elif param_id == 3:  # strideHeight
            self.RL_table_stride_height.learn(str(observation), action, reward, str(observation_))
        else:  # strideWidth
            self.RL_table_stride_width.learn(str(observation), action, reward, str(observation_))

