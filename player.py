import torch
import torch.nn.functional as F
from torch.autograd import Variable

from enviroment import Environment


class Player:
    def __init__(self, model, env, state=Environment.default_states):
        self.model = model
        self.env = env
        self.state = state

        self.hx = torch.zeros((1, 5)).cuda()
        self.cx = torch.zeros((1, 5)).cuda()

        self.reward = 0
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []

    def action_train(self):
        value, logit, (self.hx, self.cx) = self.model(
            self.state.unsqueeze(0), self.hx, self.cx
        )

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(prob, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)

        action = prob.multinomial(1).data                   # choose action
        log_prob_ = log_prob.gather(1, action)     # get corresponding prob

        self.state, self.reward = self.env.step(action.cpu().numpy()[0][0])

        self.values.append(value)
        self.log_probs.append(log_prob_)
        self.rewards.append(self.reward)

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
