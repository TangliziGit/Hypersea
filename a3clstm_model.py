import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import Config

ctx_dim = (math.ceil(Config.IMAGE_WIDTH / 4) ** 2, 512)  # (32, 512)

w_dim = 5
lstm_dim = 5


class A3C_LSTM(torch.nn.Module):
    def __init__(self, num_outputs):
        super(A3C_LSTM, self).__init__()
        self.Wai = nn.Linear(w_dim, w_dim, bias=False)
        self.Wh = nn.Linear(lstm_dim, w_dim, bias=False)
        self.att = nn.Linear(w_dim, 5)

        self.tanh = nn.Tanh()

        self.lstm = nn.LSTMCell(w_dim, lstm_dim)
        self.critic_linear = nn.Linear(lstm_dim, 1)
        self.actor_linear = nn.Linear(lstm_dim, num_outputs)

        # init weights
        self.apply(weights_init)

        self.Wai.weight.data = norm_col_init(self.Wai.weight.data, 1.0)

        self.Wh.weight.data = norm_col_init(self.Wh.weight.data, 1.0)

        self.att.weight.data = norm_col_init(self.att.weight.data, 1.0)
        self.att.bias.data.fill_(0)

        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs, hx, cx):
        # inputs: [1, 5] (B == 1)

        # attention !!!!
        # .clone() means states will be used in other way           (cause inplace op)
        # .detach() means it has no relationship with previous net  (cause zero grad on Wai, Wh)
        self.inputs = inputs.clone()
        self.inputs[0] = self.inputs[0] * 0.01
        self.Uv = self.Wai(self.inputs)                     # [1, dim]
        self.Uh = self.Wh(hx)                               # [1, dim]

        self.Uhv = self.Uv + self.Uh                        # [1, dim]
        self.TUhv = torch.tanh(self.Uhv)
        self.att_ = self.att(self.TUhv)                     # [1, 5]

        # dim=1 means softmax on 1 dim      (cause zero grad on att weight)
        self.alpha = F.softmax(self.att_, dim=1)            # [1, 5]
        self.zt = self.inputs * self.alpha                  # [1, 5 == dim]

        hx, cx = self.lstm(self.zt, (hx, cx))

        return self.critic_linear(hx), self.actor_linear(hx), (hx, cx)


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
