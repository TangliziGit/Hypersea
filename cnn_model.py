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
            nn.Conv2d(in_dim, 64, kernel_size=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 256, kernel_size=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, out_dim, kernel_size=(filter_height, filter_width), stride=(stride_height, stride_width)),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        height = ((8 - filter_height) // stride_height + 1) // 2
        width = ((8 - filter_width) // stride_width + 1) // 2
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(out_dim * height * width, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    @staticmethod
    def from_config(in_dim):
        return CnnModel(in_dim, Config.of_filter,
                        Config.filter_height, Config.filter_width,
                        Config.stride_height, Config.stride_width)

    def forward(self, inputs):
        x = self.layer(inputs)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
