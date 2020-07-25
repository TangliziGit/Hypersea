import torch


class Config:

    N_RANGE = 5

    v_t = torch.tensor([[10, 3, 5, 2, 1]]).to(torch.device('cuda')).float()
    a_t = torch.zeros(1, 5, dtype=torch.float, device=torch.device('cuda'), requires_grad=True)
    last_a_t = torch.zeros(1, 5, dtype=torch.float, device=torch.device('cuda'), requires_grad=True)
    c_t = torch.zeros(1, 64, dtype=torch.float, device=torch.device('cuda'), requires_grad=True)
    h_t = torch.zeros(1, 64, dtype=torch.float, device=torch.device('cuda'), requires_grad=True)
    last_accuracy = 0.5
    init_accuracy = 0.5
    reward = 0.00

    of_filter = 10
    filter_height = 3
    filter_width = 5
    stride_height = 2
    stride_width = 1

    best_acc = 0.5
    best_of_filter = 0
    best_filter_height = 0
    best_filter_width = 0
    best_stride_height = 0
    best_stride_width = 0

    worst_acc = 0.5
    worst_of_filter = 0
    worst_filter_height = 0
    worst_filter_width = 0
    worst_stride_height = 0
    worst_stride_width = 0

    @staticmethod
    def update_acc(acc):
        if acc > Config.best_acc:
            Config.best_of_filter = Config.of_filter
            Config.best_filter_height = Config.filter_height
            Config.best_filter_width = Config.filter_width
            Config.best_stride_height = Config.stride_height
            Config.best_stride_width = Config.stride_width
            Config.best_acc = acc

        if acc < Config.worst_acc:
            Config.worst_of_filter = Config.of_filter
            Config.worst_filter_height = Config.filter_height
            Config.worst_filter_width = Config.filter_width
            Config.worst_stride_height = Config.stride_height
            Config.worst_stride_width = Config.stride_width
            Config.worst_acc = acc

