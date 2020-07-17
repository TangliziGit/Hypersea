import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.optim as optim
from controller import Controller
from config import Config
from reforecementCNN import ReforcementQTable

five_params_QTables = ReforcementQTable()  # 创建五个对象的Q表
#
# best_acc = 0.0
# best_of_filter = 0
# best_filter_height = 0
# best_filter_width = 0
# best_stride_height = 0
# best_stride_width = 0
#
# worst_acc = 50.0
# worst_of_filter = 0
# worst_filter_height = 0
# worst_filter_width = 0
# worst_stride_height = 0
# worst_stride_width = 0


def main():
    # transform = transforms.Compose(
    #     [transforms.Resize([32, 32]),
    #      transforms.CenterCrop(32),
    #      transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = ImageFolder(root='./data/root/', transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    # testset = ImageFolder(root='./data/testRoot/', transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)


    device=torch.device('cuda')

    i=0
    while (True):
        # if(i<100):
        #     five_params_QTables.RL_table_of_filter.setEpsilon(0.7)
        #     five_params_QTables.RL_table_filter_height.setEpsilon(0.7)
        #     five_params_QTables.RL_table_filter_width.setEpsilon(0.7)
        #     five_params_QTables.RL_table_stride_height.setEpsilon(0.7)
        #     five_params_QTables.RL_table_stride_width.setEpsilon(0.7)
        # elif(99<i<200):
        #     five_params_QTables.RL_table_of_filter.setEpsilon(0.8)
        #     five_params_QTables.RL_table_filter_height.setEpsilon(0.8)
        #     five_params_QTables.RL_table_filter_width.setEpsilon(0.8)
        #     five_params_QTables.RL_table_stride_height.setEpsilon(0.8)
        #     five_params_QTables.RL_table_stride_width.setEpsilon(0.8)
        # else:
        #     five_params_QTables.RL_table_of_filter.setEpsilon(0.9)
        #     five_params_QTables.RL_table_filter_height.setEpsilon(0.9)
        #     five_params_QTables.RL_table_filter_width.setEpsilon(0.9)
        #     five_params_QTables.RL_table_stride_height.setEpsilon(0.9)
        #     five_params_QTables.RL_table_stride_width.setEpsilon(0.9)
        reward = get_Reward(device, trainloader, testloader)  # 获取奖励值
        print('reward: ',reward)
        print("--------------------------One cycle complete--------------------------The cycle number: ",i )
        if (Config.last_accuracy > 80):  # 假如说准确率达到了我们的要求
            print("of_Filter: ", Config.of_filter, " Fiter_Width: ", Config.filter_width, " Fiter_Height: ",
                  Config.filter_height,
                  " Stride_Width: ", Config.stride_width, " Stride_Height: ", Config.stride_height)
            break
        i+=1

def get_Reward(device,trainloader,testloader):#传入CIFAR数据集，对数据集进行训练后对模型进行测试后获取准确率，与上次的准确率进行比较计算奖励值送至强化学习
    # 训练RNN获取模型
    observation, action, observation_=trainRNN_Get_Model(device,trainloader)
    # 获取当前模型的准确率
    accurancy = testModel_Get_Accurancy(device,testloader)
    if(accurancy > Config.best_acc):
        Config.best_of_filter=Config.of_filter
        Config.best_filter_height=Config.filter_height
        Config.best_filter_width=Config.filter_width
        Config.best_stride_height=Config.stride_height
        Config.best_stride_width=Config.stride_width
        Config.best_acc=accurancy
    if(accurancy < Config.worst_acc):
        Config.worst_of_filter = Config.of_filter
        Config.worst_filter_height = Config.filter_height
        Config.worst_filter_width = Config.filter_width
        Config.worst_stride_height = Config.stride_height
        Config.worst_stride_width = Config.stride_width
        Config.worst_acc = accurancy
    last_acc = Config.last_accuracy
    if accurancy > last_acc:
        reward = float(accurancy - Config.init_accuracy) / 100
    else:
        reward = float(accurancy - last_acc) / 100
    Config.last_accuracy = accurancy
    five_params_QTables.learnQTable(reward, observation, action, observation_)  # 更新Q表
    Config.last_a_t = Config.a_t
    return reward

def trainRNN_Get_Model(device,trainloader):#将当前RNN网络在CIFAR数据集上训练，返回训练好的模型，训练次数待定
    observation, action, observation_ = five_params_QTables.reforcement_onestep_CNN()  # 将奖励值送进Q表内强化学习更新五个参数值
    print("CNN Params: ", "of_Filter: ", Config.of_filter, " Fiter_Height: ", Config.filter_height, " Fiter_Width: ",
          Config.filter_width, " Stride_Height: ", Config.stride_height, " Stride_Width: ", Config.stride_width)

    print("Best CNN Params: ", "of_Filter: ", Config.best_of_filter, " Fiter_Height: ", Config.best_filter_height, " Fiter_Width: ",
          Config. best_filter_width, " Stride_Height: ", Config.best_stride_height, " Stride_Width: ", Config.best_stride_width, " Best Acc: ", Config.best_acc)

    print("Worst CNN Params: ", "of_Filter: ", Config.worst_of_filter, " Fiter_Height: ",Config.worst_filter_height, " Fiter_Width: ",
          Config.worst_filter_width, " Stride_Height: ", Config.worst_stride_height, " Stride_Width: ", Config.worst_stride_width, " Worst Acc: ", Config.worst_acc)

    #初始化模型
    RnnModel=Controller(3,Config.of_filter,Config.filter_height,Config.filter_width,Config.stride_height,Config.stride_width).to(device)

    # print(RnnModel.W_h.weight)
    After_Attention_LSTM_input = RnnModel.attention()  # 做attention
    RnnModel.one_Step_In_Train_LSTM(After_Attention_LSTM_input)
    #损失函数
    criterion=nn.CrossEntropyLoss()
    optimizer = optim.SGD(RnnModel.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(10):
        totalLoss=0.0
        for i, data in enumerate(trainloader,0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = RnnModel(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
            if i % 200== 199:
                print('[Epoch %d, Batch %5d] loss: %.3f' %
                      (epoch + 1, i + 1, totalLoss / 200))
                totalLoss = 0.0
    #保存网络模型 保存整个模型
    torch.save(RnnModel, 'model.pkl')
    print("train done!")
    return observation, action, observation_

def testModel_Get_Accurancy(device,testloader):
    #获取训练好的RNN模型
    RnnModel=torch.load('model.pkl')
    correct=0
    total=0
    testData=iter(testloader)
    with torch.no_grad():
        for images, labels in testData:
            images, labels = images.to(device), labels.to(device)
            outputs = RnnModel(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accurancy=float(100 * correct / total)
    print('Accuracy of the self.network on the 10000 test images: %d %%' % accurancy)
    print('Testing Done!')
    return accurancy

if __name__ == "__main__":
    main()