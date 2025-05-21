import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from Unet.Network_test import U_Net as FCNNet

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





# 数据地址
dataset_dir = r"dataset/entropy_dataset.npy"
dataset_lable_dir = r"dataset/lables_set.npy"


dataset = np.load(dataset_dir)
dataset_lable = np.load(dataset_lable_dir)


train_on_gpu = torch.cuda.is_available()

transform = transforms.Compose([
    transforms.ToTensor(),
    ])


class Entropy_Dataset(Dataset):
    def __init__(self, dataset, lables):
        self.list_data_path = dataset
        self.list_data_label = lables
        self.transform = transform

    def __len__(self):
        return len(self.list_data_path)

    def __getitem__(self, idx):
        data = np.load(self.list_data_path[idx], allow_pickle=True)
        label = np.load(self.list_data_label[idx], allow_pickle=True)
        return data, label.reshape(1,84)


# 测试集、训练集、验证集。
indices = list(range(len(dataset)))
np.random.shuffle(indices)  # 打乱顺序

train_idx = indices[: int(len(dataset)*0.8)]
valid_idx = indices[int(len(dataset)*0.8): int(len(dataset)*0.9)]
test_idx = indices[int(len(dataset)*0.9):]



train_sampler = SubsetRandomSampler(train_idx)  # 确定采样的顺序，后面制作train_loader的时用这个列表得索引值取样本
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

data_set = Entropy_Dataset(dataset, dataset_lable)


train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,drop_last=True)
valid_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers,drop_last=True)
test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers,drop_last=True)


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
    all_log_probs = []
    all_labels = []

    with torch.no_grad():
        for data, target in data_loader:
            # 数据转移
            data = data.float().to(device)
            target = target.float().to(device).squeeze(1)  # [batch_size, 84]

            # 模型预测
            logits = model(data)
            log_probs = F.log_softmax(logits, dim=1)

            # 保存结果
            all_log_probs.append(log_probs)
            all_labels.append(target)

    # 合并结果
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return calculate_metrics(all_log_probs, all_labels)


# 训练和评估的主函数
def train_and_evaluate(model, train_loader, valid_loader, test_loader, criterion, optimizer, device, num_epochs):
    """
    训练模型并评估其性能

    参数:
        model (torch.nn.Module): 模型
        train_loader (torch.utils.data.DataLoader): 训练数据加载器
        valid_loader (torch.utils.data.DataLoader): 验证数据加载器
        test_loader (torch.utils.data.DataLoader): 测试数据加载器
        criterion (torch.nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        device (torch.device): 设备（CPU 或 GPU）
        num_epochs (int): 训练的总 epoch 数
    """
    best_kl_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        for step, (data, target) in enumerate(train_loader):
            # 数据准备
            data = data.float().to(device)
            target = target.float().to(device).squeeze(1)  # [batch_size, 84]

            # 前向传播
            logits = model(data)
            log_probs = F.log_softmax(logits, dim=1)

            # 计算损失
            loss = criterion(log_probs, target)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch: {epoch + 1}, Step: {step}, Loss: {loss.item():.4f}")

        # 验证阶段
        valid_metrics = evaluate_model(model, valid_loader, device)
        print(f"Epoch: {epoch + 1} Validation || "
              f"KL Loss: {valid_metrics['kl_loss']:.4f} | "
              f"CE Loss: {valid_metrics['ce_loss']:.4f} | "
              f"Brier: {valid_metrics['brier_score']:.4f} | "
              f"CosSim: {valid_metrics['cos_sim']:.4f}")

        # 保存最佳模型
        if valid_metrics["kl_loss"] < best_kl_loss:
            best_kl_loss = valid_metrics["kl_loss"]
            best_model_state = model.state_dict()
            torch.save(best_model_state, "best_entropy_model.pth")
            print(f"New best model saved with KL Loss: {best_kl_loss:.4f}")

    # 最终测试
    model.load_state_dict(torch.load("best_entropy_model.pth"))
    test_metrics = evaluate_model(model, test_loader, device)
    print("\nFinal Test Metrics:")
    print(f"KL Loss: {test_metrics['kl_loss']:.4f} | "
          f"CE Loss: {test_metrics['ce_loss']:.4f} | "
          f"Brier: {test_metrics['brier_score']:.4f} | "
          f"CosSim: {test_metrics['cos_sim']:.4f}")


if __name__ == '__main__':
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型初始化（需定义FCNNet类）
    contact_net = FCNNet(entropy_ch=3).to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(contact_net.parameters(), lr=1e-3)  # lr=1e-3
    criterion = nn.KLDivLoss(reduction='batchmean')


    # 运行训练
    train_and_evaluate(
        contact_net,
        train_loader,
        valid_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        total_epochs
    )

