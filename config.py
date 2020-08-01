import torch


class Config:
    TEST = False
    N_RANGE = 75

    config_bak = {}

    v_t = torch.tensor([[10, 3, 5, 2, 1]]).to(torch.device('cuda')).float()
    a_t = torch.zeros(1, 5, dtype=torch.float, device=torch.device('cuda'), requires_grad=True)
    last_a_t = torch.zeros(1, 5, dtype=torch.float, device=torch.device('cuda'), requires_grad=True)
    # c_t = torch.zeros(1, 64, dtype=torch.float, device=torch.device('cuda'), requires_grad=True)
    # h_t = torch.zeros(1, 64, dtype=torch.float, device=torch.device('cuda'), requires_grad=True)
    h_t = None
    c_t = None
    last_accuracy = 0.5
    reward = 0.00

    # VGG3: [512, 2, 2, 1, 1]
    of_filter = 512
    filter_height = 3
    filter_width = 5
    stride_height = 2
    stride_width = 1

    best_acc = 0.0
    best_of_filter = 0
    best_filter_height = 0
    best_filter_width = 0
    best_stride_height = 0
    best_stride_width = 0

    worst_acc = 1.0
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

    @staticmethod
    def backup():
        cb = Config.config_bak

        cb['v_t'], cb['a_t'], cb['last_a_t'], cb['h_t'], cb['c_t'] = \
            Config.v_t, Config.a_t, Config.last_a_t, Config.h_t, Config.c_t

        cb['last_accuracy'], cb['reward'] = Config.last_accuracy, Config.reward

        cb['of_filter'], cb['filter_height'], cb['filter_width'], cb['stride_height'], cb['stride_width'] = \
            Config.of_filter, Config.filter_height, Config.filter_width, Config.stride_height, Config.stride_width

        # cb['best_of_filter'], cb['best_filter_height'], cb['best_filter_width'], cb['best_stride_height'],\
        #     cb['best_stride_width'], cb['best_acc'] = \
        #    Config.best_of_filter, Config.best_filter_height, Config.best_filter_width, Config.best_stride_height,\
        #    Config.best_stride_width, Config.best_acc

        # cb['worst_of_filter'], cb['worst_filter_height'], cb['worst_filter_width'], cb['worst_stride_height'],\
        #     cb['worst_stride_width'], cb['worst_acc'] = \
        #     Config.worst_of_filter, Config.worst_filter_height, Config.worst_filter_width, Config.worst_stride_height,\
        #     Config.worst_stride_width, Config.worst_acc

    @staticmethod
    def rollback():
        cb = Config.config_bak

        Config.v_t, Config.a_t, Config.last_a_t, Config.h_t, Config.c_t = \
            cb['v_t'], cb['a_t'], cb['last_a_t'], cb['h_t'], cb['c_t']

        Config.last_accuracy, Config.reward = \
            cb['last_accuracy'], cb['reward']

        Config.of_filter, Config.filter_height, Config.filter_width, Config.stride_height, Config.stride_width = \
            cb['of_filter'], cb['filter_height'], cb['filter_width'], cb['stride_height'], cb['stride_width']

        # Config.best_of_filter, Config.best_filter_height, Config.best_filter_width, Config.best_stride_height,\
        #     Config.best_stride_width, Config.best_acc = \
        #     cb['best_of_filter'], cb['best_filter_height'], cb['best_filter_width'], cb['best_stride_height'], cb[
        #         'best_stride_width'], cb['best_acc']

        # Config.worst_of_filter, Config.worst_filter_height, Config.worst_filter_width, Config.worst_stride_height,\
        #     Config.worst_stride_width, Config.worst_acc = \
        #     cb['worst_of_filter'], cb['worst_filter_height'], cb['worst_filter_width'], cb['worst_stride_height'], cb[
        #         'worst_stride_width'], cb['worst_acc']

