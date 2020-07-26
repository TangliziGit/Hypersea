import torch
import torch.nn as nn
from config import Config
from logger import Logger


class CnnModel(nn.Module):
    def __init__(self, in_dim, out_dim=Config.of_filter,
                 filter_height=Config.filter_height, filter_width=Config.filter_width,
                 stride_height=Config.stride_height, stride_width=Config.stride_width):
        super(CnnModel, self).__init__()
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
            nn.ReLU(inplace=True),
            nn.Conv2d(130, 270, kernel_size=(4, 5), stride=(2, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(270, out_dim, kernel_size=(filter_height, filter_width), stride=(stride_height, stride_width)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )

        self.fc = nn.Sequential(
            nn.Linear(out_dim * int(((14 - filter_height) / stride_height + 1) / 2) *
                      int(((24 - filter_width) / stride_width + 1) / 2), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    @staticmethod
    def from_config(in_dim):
        return CnnModel(in_dim, Config.of_filter,
                        Config.filter_height, Config.filter_width,
                        Config.stride_height, Config.stride_width)

    def forward(self, inputs):

        self.out_dim = Config.of_filter
        self.filter_height = Config.filter_height
        self.filter_width = Config.filter_width
        self.stride_height = Config.stride_height
        self.stride_width = Config.stride_width

        y = self.layer(inputs)
        y = y.view(-1, self.out_dim * int(((14 - self.filter_height) / self.stride_height + 1) / 2) *
                    int(((24 - self.filter_width) / self.stride_width + 1) / 2))
        y = self.fc(y)
        return y

