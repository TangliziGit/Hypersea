import torch
import torch.nn as nn
from config import Config


class Controller(nn.Module):
    def __init__(self, in_dim, out_dim=Config.of_filter,
                 filter_height=Config.filter_height, filter_width=Config.filter_width,
                 stride_height=Config.stride_height, stride_width=Config.stride_width):
        super(Controller, self).__init__()
        self.DEVICE = torch.device('cuda')
        self.input_size = 5
        self.hidden_size = 64

        self.lstmCell = torch.nn.LSTMCell(self.input_size, self.hidden_size).to(self.DEVICE)
        self.LSTMInitFlag = False

        self.W_h = nn.Linear(64, 64, bias=False).to(self.DEVICE)  # 进行注意力机制时LSTMCell上的权重
        self.W_v = nn.Linear(5, 64, bias=False).to(self.DEVICE)
        self.W_g = nn.Linear(64, 5, bias=False).to(self.DEVICE)

        self.out_dim = out_dim
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride_height = stride_height
        self.stride_width = stride_width

        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, 130, kernel_size=(3, 5), stride=(1, 1)),
            nn.Conv2d(130, 270, kernel_size=(4, 5), stride=(2, 1)),
            nn.Conv2d(270, out_dim, kernel_size=(filter_height, filter_width), stride=(stride_height, stride_width)),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )
        self.fc1 = nn.Linear(out_dim * int(((14 - filter_height) / stride_height + 1) / 2) *
                             int(((24 - filter_width) / stride_width + 1) / 2), 100)
        self.fc2 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(p=0.5)

    @staticmethod
    def from_config(in_dim):
        return Controller(in_dim, Config.of_filter,
                          Config.filter_height, Config.filter_width,
                          Config.stride_height, Config.stride_width)

    def forward(self, inputx):

        self.out_dim = Config.of_filter
        self.filter_height = Config.filter_height
        self.filter_width = Config.filter_width
        self.stride_height = Config.stride_height
        self.stride_width = Config.stride_width

        y = self.layer(inputx)
        y1 = y.view(-1, self.out_dim * int(((14 - self.filter_height) / self.stride_height + 1) / 2) *
                    int(((24 - self.filter_width) / self.stride_width + 1) / 2))
        y2 = self.fc1(y1)
        y3 = self.dropout(y2)
        y4 = self.fc2(y3)
        return y4

    def attention(self):  # 一步注意力机制，从Config中获取v_t与h_t,经过注意力机制后输出z_t送予LSTM，将a_t存储在Config中
        h_t = Config.h_t  # [1*64]
        v_t = torch.tensor([[Config.of_filter, Config.filter_height, Config.filter_width, Config.stride_height,
                             Config.stride_width]]).to(self.DEVICE).float()  # [1*5]
        # 注意力机制第一步
        h_t = self.W_h(h_t)  # [1*64]*[64*64]=[1*64]
        v_t = self.W_v(v_t)  # [1*5]*[5*64]=[1*64]
        # 两个数据乘以相应权重后用tanh函数函数取值后存储在g_t
        g_t = torch.tanh(input=(h_t + v_t)).to(self.DEVICE)  # [1*64]
        # 注意力机制第二步
        s_t = (self.W_g(g_t) + 0.01).to(self.DEVICE)  # [1*64]*[64*5]=[1*5]
        # 注意力机制第三步
        a_t = torch.softmax(s_t, 1).to(self.DEVICE)  # [1*5]
        # 将a_t存储在Config中，为了后续强化学习时选取需要进行学习的参数
        Config.a_t = a_t  # [1*5]
        v_t = torch.tensor([[Config.of_filter, Config.filter_height, Config.filter_width, Config.stride_height,
                             Config.stride_width]]).to(self.DEVICE).float()

        numpya_t = a_t.cpu().detach().numpy()
        numpyv_t = v_t.cpu().detach().numpy()
        # 注意力机制第四步
        test1 = numpya_t[0][0] * numpyv_t[0][0]
        test2 = numpya_t[0][1] * numpyv_t[0][1]
        test3 = numpya_t[0][2] * numpyv_t[0][2]
        test4 = numpya_t[0][3] * numpyv_t[0][3]
        test5 = numpya_t[0][4] * numpyv_t[0][4]
        z_t = torch.tensor([[test1, test2, test3, test4, test5]]).to(self.DEVICE)  # [1*5]
        print("Attention Done!")
        return z_t

    def one_Step_In_Train_LSTM(self, After_Attention_LSTM_input):  # 一步LSTM输入输出，将h_t与c_t存储在Config中
        if not self.LSTMInitFlag:
            Config.h_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.DEVICE)
            Config.c_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.DEVICE)
            self.LSTMInitFlag = True
        Config.h_t, Config.c_t = self.lstmCell(After_Attention_LSTM_input, (Config.h_t, Config.c_t))
        print('LSTM Done!')
