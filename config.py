class Config:
    TEST = True
    N_CNN_EPOCH = 0 # 75
    N_STEPS = 4
    IMAGE_WIDTH = 32

    lr = 0.001
    gamma = 0.99
    tau = 1.00

    init_states = [512, 3, 5, 2, 2]

    acc = 0.0
    states = init_states

    best_acc = 0.0
    best_states = [0, 0, 0, 0, 0]

    worst_acc = 1.0
    worst_states = [0, 0, 0, 0, 0]

    @staticmethod
    def update_states(acc, states):
        Config.acc, Config.states = acc, states

        if acc > Config.best_acc:
            Config.best_states = states
            Config.best_acc = acc

        if acc < Config.worst_acc:
            Config.worst_states = states
            Config.worst_acc = acc

