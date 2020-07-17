import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from controller import Controller
from config import Config
from reforecementCNN import ReinforcementQTable

five_params_QTables = ReinforcementQTable()  # 创建五个对象的Q表


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device('cuda')

    i = 0
    while True:
        reward = get_Reward(device, train_loader, test_loader)  # 获取奖励值
        print(f'reward: {reward}')
        print("--------------------------One cycle complete--------------------------"
              f"The cycle number: {i}")
        
        if Config.last_accuracy > 80:  
            # 假如说准确率达到了我们的要求
            print(f"of_Filter: {Config.of_filter}, "
                  f"Filter_Width: {Config.filter_width}, Filter_Height: {Config.filter_height}, "
                  f"Stride_Width: {Config.stride_width}, Stride_Height: {Config.stride_height}")
            break
        i += 1


def get_Reward(device, train_loader, test_loader):  
    # 传入CIFAR数据集，对数据集进行训练后对模型进行测试后获取准确率，与上次的准确率进行比较计算奖励值送至强化学习
    
    # 训练RNN获取模型
    observation, action, observation_ = trainRNN_Get_Model(device, train_loader)
    
    # 获取当前模型的准确率
    accuracy = testModel_Get_Accurancy(device, test_loader)
    
    if accuracy > Config.best_acc:
        Config.best_of_filter = Config.of_filter
        Config.best_filter_height = Config.filter_height
        Config.best_filter_width = Config.filter_width
        Config.best_stride_height = Config.stride_height
        Config.best_stride_width = Config.stride_width
        Config.best_acc = accuracy
        
    if accuracy < Config.worst_acc:
        Config.worst_of_filter = Config.of_filter
        Config.worst_filter_height = Config.filter_height
        Config.worst_filter_width = Config.filter_width
        Config.worst_stride_height = Config.stride_height
        Config.worst_stride_width = Config.stride_width
        Config.worst_acc = accuracy
        
    last_acc = Config.last_accuracy
    
    if accuracy > last_acc:
        reward = float(accuracy - Config.init_accuracy) / 100
    else:
        reward = float(accuracy - last_acc) / 100
        
    Config.last_accuracy = accuracy
    five_params_QTables.learnQTable(reward, observation, action, observation_)  # 更新Q表
    Config.last_a_t = Config.a_t
    
    return reward


def trainRNN_Get_Model(device, train_loader):
    # 将当前RNN网络在CIFAR数据集上训练，返回训练好的模型，训练次数待定
    
    observation, action, observation_ = five_params_QTables.reinforcement_one_step_cnn()  # 将奖励值送进Q表内强化学习更新五个参数值
    
    print(f"CNN Params: of_Filter: {Config.of_filter}, "
          f"Filter_Height: {Config.filter_height}, Filter_Width: {Config.filter_width}, "
          f"Stride_Height: {Config.stride_height}, Stride_Width: {Config.stride_width}")

    print(f"Best CNN Params: of_Filter: {Config.best_of_filter}, "
          f"Fiter_Height: {Config.best_filter_height}, Fiter_Width: {Config.best_filter_width}, "
          f"Stride_Height: {Config.best_stride_height}, Stride_Width: {Config.best_stride_width}, "
          f"Best Acc: {Config.best_acc}")

    print(f"Worst CNN Params: of_Filter: {Config.worst_of_filter}, "
          f"Fiter_Height: {Config.worst_filter_height}, Fiter_Width: {Config.worst_filter_width}, "
          f"Stride_Height: {Config.worst_stride_height}, Stride_Width: {Config.worst_stride_width}, "
          f"Worst Acc: {Config.worst_acc}")

    # 初始化模型
    RnnModel = Controller.from_config(3).to(device)

    After_Attention_LSTM_input = RnnModel.attention()  # 做attention
    RnnModel.one_Step_In_Train_LSTM(After_Attention_LSTM_input)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(RnnModel.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(10):
        totalLoss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = RnnModel(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
            if i % 200 == 199:
                print('[Epoch %d, Batch %5d] loss: %.3f' %
                      (epoch + 1, i + 1, totalLoss / 200))
                totalLoss = 0.0
                
    # 保存网络模型 保存整个模型
    torch.save(RnnModel, 'model.pkl')
    print("train done!")
    return observation, action, observation_


def testModel_Get_Accurancy(device, test_loader):
    # 获取训练好的RNN模型
    
    RnnModel = torch.load('model.pkl')
    correct = 0
    total = 0
    testData = iter(test_loader)
    with torch.no_grad():
        for images, labels in testData:
            images, labels = images.to(device), labels.to(device)
            outputs = RnnModel(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = float(100 * correct / total)
    print('Accuracy of the self.network on the 10000 test images: %d %%' % accuracy)
    print('Testing Done!')
    return accuracy


if __name__ == "__main__":
    main()
