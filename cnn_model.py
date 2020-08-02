import torch
import torch.nn as nn
from config import Config
from logger import Logger


class CnnModel(nn.Module):
    def __init__(self, in_dim, out_dim, filter_height, filter_width, stride_height, stride_width):
        super(CnnModel, self).__init__()
        self.DEVICE = torch.device('cuda')

        self.out_dim = out_dim
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride_height = stride_height
        self.stride_width = stride_width

        self.conv = nn.Sequential(
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
    def from_states(states, in_dim=3):
        states_ = states.cpu().numpy().astype(int)
        return CnnModel(in_dim, *states_)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
