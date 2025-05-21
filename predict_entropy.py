import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from Unet.Network_test import U_Net as FCNNet
import math

#
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())



# 设置学习次数
total_epochs = 200
# number of subprocesses to use for data loading
num_workers = 8
# 采样批次
batch_size = 70
# 学习率
lr = 0.001




# 数据地址
dataset_dir = r"dataset/entropy_dataset_predict.npy"



dataset = np.load(dataset_dir)#[:32]



train_on_gpu = torch.cuda.is_available()



class Entropy_Dataset(Dataset):
    def __init__(self, dataset):
        self.list_data_path = dataset

    def __len__(self):
        return len(self.list_data_path)

    def __getitem__(self, idx):
        data = np.load(self.list_data_path[idx], allow_pickle=True)
        data = data[0:2, :, :]  # 选择维度
        return data


data_set = Entropy_Dataset(dataset)


predict_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, num_workers=num_workers)


# 计算所有评估指标的函数
def calculate_metrics(log_probs, labels):
    """
    计算所有评估指标（适用于概率分布标签）

    参数:
        log_probs (torch.Tensor): 模型的对数概率输出，形状为 (batch_size, 84)
        labels (torch.Tensor): 真实的概率分布标签，形状为 (batch_size, 84)

    返回:
        metrics (dict): 包含所有评估指标的字典
    """
    # KL散度（与训练损失一致）
    kl_loss = F.kl_div(log_probs, labels, reduction='batchmean')    #  KL散度的值总是非负的（≥ 0），并且在两个分布完全相同的情况下为0。值越小，表示两个分布越相似,希望最小化KL散度损失。

    # 交叉熵损失（使用对数概率计算）
    ce_loss = -torch.sum(labels * log_probs) / labels.size(0)       #  交叉熵损失的值也是非负的（≥ 0），且在模型预测完全正确时为0。值越小，表示模型性能越好。

    # Brier分数（需要概率值）
    probs = log_probs.exp()
    brier_score = torch.mean(torch.sum((probs - labels) ** 2, dim=1))  # Brier分数的值在0到1之间，0表示完美的预测，1表示最差的预测。 希望最小化Brier分数。

    # 余弦相似度
    cos_sim = F.cosine_similarity(probs, labels, dim=1).mean()  # 余弦相似度的值在-1到1之间，1表示完全相同，0表示无相似度，-1表示完全相反。希望最大化余弦相似度。

    return {
        "kl_loss": kl_loss.item(),
        "ce_loss": ce_loss.item(),
        "brier_score": brier_score.item(),
        "cos_sim": cos_sim.item()
    }


# 评估整个数据集的函数
def evaluate_model(model, data_loader, device):
    """
    评估模型在整个数据集上的性能

    参数:
        model (torch.nn.Module): 训练好的模型
        data_loader (torch.utils.data.DataLoader): 数据加载器
        device (torch.device): 设备（CPU 或 GPU）

    返回:
        metrics (dict): 包含所有评估指标的字典
    """
    model.eval()
    all_predict = []
    all_log_probs = []

    with torch.no_grad():
        for data in data_loader:
            # 数据转移
            data = data.float().to(device)

            # 模型预测
            logits = model(data)
            log_probs = F.log_softmax(logits, dim=1)
            # log_probs = log_probs.exp()
            all_log_probs.append(log_probs.to("cpu"))

        #all_predict.append(-sum([d * math.log2(d) for d in log_probs if d > 0]))

    return all_log_probs



if __name__ == '__main__':
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型初始化（需定义FCNNet类）
    contact_net = FCNNet(entropy_ch=2).to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(contact_net.parameters(), lr=1e-3)  # lr=1e-3
    criterion = nn.KLDivLoss(reduction='batchmean')

    model = contact_net

    model.load_state_dict(torch.load("best_model_HB+BB.pth"))
    entropy_predict = evaluate_model(model, predict_loader, device)
    print("预测完毕，开始后处理")
    all_predict = []
    for i in range(len(entropy_predict)):
        print(i)
        for j in range(len(entropy_predict[i])):
            all_predict.append(-sum([d * math.log2(d) for d in entropy_predict[i][j].exp() if d > 0]))

    # for i in range(len(all_predict)):
    #     print(all_predict[i].item())

    with open('shannon_entropy_predict.txt', 'w') as file:
        for i in range(len(all_predict)):
            file.write(f"{all_predict[i].item()}\n")  # 每个元素占一行
    print("制作完毕!")



