import torch
import torch.nn as nn

from config import Config
from logger import Logger

class ControllerModel(nn.Module):
    def __init__(self):
        super(ControllerModel, self).__init__()
        self.device = torch.device("cuda")
        self.input_size = 5
        self.hidden_size = 64
        self.lstmCell = torch.nn.LSTMCell(self.input_size, self.hidden_size).to(self.device)

        # 进行注意力机制时LSTMCell上的权重
        self.W_h = nn.Linear(64, 64, bias=False).to(self.device)
        self.W_v = nn.Linear(5, 64, bias=False).to(self.device)
        self.W_g = nn.Linear(64, 5, bias=False).to(self.device)

        if Config.h_t is None or Config.c_t is None:
            Config.h_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)
            Config.c_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)

    def attention(self):
        h_t = Config.h_t  # [1*64]
        v_t = torch.tensor([[
            Config.of_filter,
            Config.filter_height, Config.filter_width,
            Config.stride_height, Config.stride_width
        ]]).to(self.device).float()  # [1*5]

        # 注意力机制第一步
        h_t = self.W_h(h_t)  # [1*64]*[64*64]=[1*64]
        v_t = self.W_v(v_t)  # [1*5]*[5*64]=[1*64]
        # 两个数据乘以相应权重后用tanh函数函数取值后存储在g_t
        g_t = torch.tanh(h_t + v_t).to(self.device)  # [1*64]
        # 注意力机制第二步
        s_t = (self.W_g(g_t) + 0.01).to(self.device)  # [1*64]*[64*5]=[1*5]
        # 注意力机制第三步
        a_t = torch.softmax(s_t, 1).to(self.device)  # [1*5]

        Config.a_t = a_t  # [1*5]
        v_t = torch.tensor([[
            Config.of_filter,
            Config.filter_height, Config.filter_width,
            Config.stride_height, Config.stride_width
        ]]).to(self.device).float()

        numpya_t = a_t.cpu().detach().numpy()
        numpyv_t = v_t.cpu().detach().numpy()
        # 注意力机制第四步
        test1 = numpya_t[0][0] * numpyv_t[0][0]
        test2 = numpya_t[0][1] * numpyv_t[0][1]
        test3 = numpya_t[0][2] * numpyv_t[0][2]
        test4 = numpya_t[0][3] * numpyv_t[0][3]
        test5 = numpya_t[0][4] * numpyv_t[0][4]

        z_t = torch.tensor([[test1, test2, test3, test4, test5]]).to(self.device)

        Logger.stage("attention", f"a_t: {Config.a_t.data}")
        return z_t

    def lstm(self, attention):
        Config.h_t, Config.c_t = self.lstmCell(attention, (Config.h_t, Config.c_t))

        Logger.stage("lstm", f"done")
