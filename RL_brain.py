import numpy as np
import pandas as pd
from reforcementStatus import ReforcementStatus
import math
import copy

class QLearningTable:
    def __init__(self, actions, e_greedy=0.9,learning_rate=0.8, reward_decay=0.98):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        state_action = copy.deepcopy(self.q_table.loc[observation, :])
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)#根据概率选择相应动作
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
    def setEpsilon(self,i):
        self.epsilon=i



