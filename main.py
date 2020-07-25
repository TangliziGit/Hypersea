import pickle as pkl
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from logger import Logger
from controller import Controller
from config import Config
from q_tables import QTables

q_tables = QTables()  # 创建五个对象的Q表
losses = []


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    device = torch.device('cuda')

    i = 0
    while Config.last_accuracy <= 0.80:
        Logger.iteration_start(i)
        reward = get_reward(device, train_loader, test_loader)

        Logger.stage('reward', f'reward: {reward}')
        i += 1

    Logger.show_params()


def get_reward(device, train_loader, test_loader):
    # 传数据集，对数据集进行训练后对模型进行测试后获取准确率，与上次的准确率进行比较计算奖励值送至强化学习

    # 选择一个参数进行更新，获取前后状态和对应动作
    observation, action, observation_ = q_tables.step()  # 将奖励值送进Q表内强化学习更新五个参数值

    Logger.show_params()

    # 训练&获取&测试模型
    model = train_cnn(device, train_loader)
    accuracy = test_cnn(model, device, test_loader)
    reward = accuracy - Config.last_accuracy

    # 检查正确率，更新正确率最值
    Config.update_acc(accuracy)
    Config.last_accuracy = accuracy

    q_tables.learn(reward, observation, action, observation_)  # 更新Q表
    Config.last_a_t = Config.a_t

    return reward


def train_cnn(device, train_loader):

    # 根据刚才更新的参数，初始化模型
    model = Controller.from_config(3).to(device)

    attention = model.attention()
    model.lstm(attention)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    log_interval = 60
    losses.append([])

    Logger.stage('train', 'start')
    for epoch in range(Config.N_RANGE):
        totalLoss = .0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()

            optimizer.step()
            totalLoss += loss.item()
            losses[-1].append(loss.item())

            if (i + 1) % log_interval == 0:
                Logger.print(f'[Epoch {epoch}, Batch {i}] loss: {totalLoss / log_interval}')
                totalLoss = .0

    Logger.stage('train', 'finish')
    pkl.dump(losses, open('losses.pkl', 'wb'))
    torch.save(model, 'model.pkl')
    return model


def test_cnn(model, device, test_loader):
    correct, total = .0, .0
    testData = iter(test_loader)
    with torch.no_grad():
        for images, labels in testData:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    Logger.stage('test', f"acc: {accuracy}")
    return accuracy


if __name__ == "__main__":
    main()
