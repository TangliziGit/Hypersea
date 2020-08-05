import traceback

import torch
from torch import optim
from torch.autograd import Variable

from a3clstm_model import A3C_LSTM
from enviroment import Environment
from logger import Logger
from config import Config

from player import Player


def main():
    iterate()
    Logger.show_status()


def iterate():
    model = A3C_LSTM(len(Environment.action_space)).cuda()
    env = Environment()
    player = Player(model, env)

    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    iter = 0
    while True:
        Logger.iteration_start(iter)
        for step in range(Config.N_STEPS):
            Logger.show_status()
            player.action_train()

        value, _, _ = player.model(player.state.unsqueeze(0), player.hx, player.cx)
        R = value.cuda()

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1).cuda()
        R = Variable(R)
        for i in reversed(range(len(player.rewards))):
            R = Config.gamma * R + player.rewards[i]
            advantage = R - player.values[i].cuda()
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + Config.gamma * \
                player.values[i + 1].cuda().data - player.values[i].cuda().data

            gae = gae * Config.gamma * Config.tau + delta_t

            policy_loss = policy_loss - \
                player.log_probs[i].cuda() * \
                Variable(gae).cuda() - 0.01 * player.entropies[i].cuda()

        player.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward(retain_graph=True)

        optimizer.step()
        player.clear_actions()
        Logger.stage('loss', f'{(policy_loss + 0.5*value_loss).item()}')
        iter += 1


if __name__ == "__main__":
    try:
        torch.autograd.set_detect_anomaly(True)
        main()
    except Exception as e:
        Logger.error(traceback.format_exc())

