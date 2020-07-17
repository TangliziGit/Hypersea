import torch
from config import Config
from RL_brain import QLearningTable
from reforcementStatus import ReforcementStatus
class ReforcementQTable:
    def __init__(self):
        # 初始化每个参数的状态空间
        self.RL_of_filter = ReforcementStatus(10, 20,0,0)
        self.RL_filter_height = ReforcementStatus(3, 1,15,1)
        self.RL_filter_width = ReforcementStatus(5, 1,25,1)
        self.RL_stride_height = ReforcementStatus(2, 1,0,0)
        self.RL_stride_width = ReforcementStatus(1, 1,0,0)
        # 对每个状态空间初始化Q表
        self.RL_table_of_filter = QLearningTable(actions=list(range(self.RL_of_filter.n_actions)))
        self.RL_table_filter_height = QLearningTable(actions=list(range(self.RL_filter_height.n_actions)))
        self.RL_table_filter_width = QLearningTable(actions=list(range(self.RL_filter_width.n_actions)))
        self.RL_table_stride_height = QLearningTable(actions=list(range(self.RL_stride_height.n_actions)))
        self.RL_table_stride_width = QLearningTable(actions=list(range(self.RL_stride_width.n_actions)))

    def reforcement_onestep_CNN(self):
        a_t = Config.a_t
        # 判断是哪个参数需要进行强化学习
        testParameterTensor = torch.max(a_t, 1)[1]
        testParameter = testParameterTensor.item()
        if (testParameter == 0):  # ofFilter
            observation = Config.of_filter
            # 做一个延迟等待操作
            self.RL_of_filter.render()
            # 在Q表中使用当前状态值去获取将要发生的动作
            action = self.RL_table_of_filter.choose_action(str(observation))
            # 使用动作去获取更新后的值
            observation_ = self.RL_of_filter.step(action)
            # 将其存在Config中等待下次调用
            Config.of_filter = observation_
            # RL learn from this transition
            return observation,action,observation_
            print("Change CNN Param Done!")
        elif (testParameter == 1):  # filterHeight
            observation = Config.filter_height
            # 做一个延迟等待操作
            self.RL_filter_height.render()
            # 在Q表中使用当前状态值去获取将要发生的动作
            action = self.RL_table_filter_height.choose_action(str(observation))
            # 使用动作去获取更新后的值
            observation_ = self.RL_filter_height.step(action)
            # 将其存在Config中等待下次调用
            Config.filter_height = observation_
            print("Change CNN Param Done!")
            return observation,action,observation_
        elif (testParameter == 2):  # filterWidth
            observation = Config.filter_width
            # 做一个延迟等待操作
            self.RL_filter_width.render()
            # 在Q表中使用当前状态值去获取将要发生的动作
            action = self.RL_table_filter_width.choose_action(str(observation))
            # 使用动作去获取更新后的值
            observation_ = self.RL_filter_width.step(action)
            # 将其存在Config中等待下次调用
            Config.filter_width = observation_
            print("Change CNN Param Done!")
            return observation,action,observation_
        elif (testParameter == 3):  # strideHeight
            observation = Config.stride_height
            # 做一个延迟等待操作
            self.RL_stride_height.render()
            # 在Q表中使用当前状态值去获取将要发生的动作
            action = self.RL_table_stride_height.choose_action(str(observation))
            # 使用动作去获取更新后的值
            observation_ = self.RL_stride_height.step(action)
            # 将其存在Config中等待下次调用
            Config.stride_height = observation_
            print("Change CNN Param Done!")
            return observation,action,observation_
        else:  # strideWidth
            observation = Config.stride_width
            # 做一个延迟等待操作
            self.RL_stride_width.render()
            # 在Q表中使用当前状态值去获取将要发生的动作
            action = self.RL_table_stride_width.choose_action(str(observation))
            # 使用动作去获取更新后的值
            observation_ = self.RL_stride_width.step(action)
            # 将其存在Config中等待下次调用
            Config.stride_width = observation_
            print("Change CNN Param Done!")
            return observation,action,observation_

    def learnQTable(self,reward,observation,action,observation_):
        a_t = Config.last_a_t
        # 判断是哪个参数需要进行强化学习
        testParameterTensor = torch.max(a_t, 1)[1]
        testParameter = testParameterTensor.item()
        if (testParameter == 0):  # ofFilter
            self.RL_table_of_filter.learn(str(observation), action, reward, str(observation_))
        elif (testParameter == 1):  # filterHeight
            self.RL_table_filter_height.learn(str(observation), action, reward, str(observation_))
        elif (testParameter == 2):  # filterWidth
            self.RL_table_filter_width.learn(str(observation), action, reward, str(observation_))
        elif (testParameter == 3):  # strideHeight
            self.RL_table_stride_height.learn(str(observation), action, reward, str(observation_))
        else: #strideWidth
            self.RL_table_stride_width.learn(str(observation), action, reward, str(observation_))
        print("Update Qtable Done!")