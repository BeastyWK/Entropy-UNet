import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from Unet.Network_0328 import U_Net as FCNNet
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec






total_epochs = 10

num_workers = 8

batch_size = 32

lr = 0.001




# 数据地址
dataset_dir = r"dataset/entropy_dataset_predict.npy"



dataset = np.load(dataset_dir)[16000:16032]#[:32]



train_on_gpu = torch.cuda.is_available()



class Entropy_Dataset(Dataset):
    def __init__(self, dataset):
        self.list_data_path = dataset

    def __len__(self):
        return len(self.list_data_path)

    def __getitem__(self, idx):
        data = np.load(self.list_data_path[idx], allow_pickle=True)
        # data = data[0:2, :, :]  # 选择维度
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


# Grad-CAM 类
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output  # 不要使用 .detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # 梯度需要分离

    def __call__(self, x, target_index=None):
        self.model.eval()  # 确保模型在评估模式
        self.model.zero_grad()

        # 前向传播（保留计算图）
        logits = self.model(x)  # 形状 [1, 84]

        # 构造目标梯度
        if target_index is None:
            target_index = logits.argmax(dim=1)
        target = torch.zeros_like(logits)
        target[0, target_index] = 1.0

        # 反向传播
        logits.backward(gradient=target, retain_graph=False)

        # 计算权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 生成热力图
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=True)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy()




if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    contact_net = FCNNet(entropy_ch=3).to(device)


    optimizer = optim.Adam(contact_net.parameters(), lr=1e-3)  # lr=1e-3
    criterion = nn.KLDivLoss(reduction='batchmean')


    contact_net.load_state_dict(torch.load("best_entropy_model_great.pth"))
    contact_net.eval()  


    target_layer = contact_net.Conv4.conv[0]  
    gradcam = GradCAM(model=contact_net, target_layer=target_layer)



    sample_data = next(iter(predict_loader))
    sample_data = sample_data.float().to(device)[0:1]  
    sample_data.requires_grad_(True) 

    with torch.enable_grad():
        cam = gradcam(sample_data, target_index=0)  








    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)


    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(3, 3,
                  width_ratios=[1, 1, 0.04],
                  wspace=0.001,
                  hspace=0.25,
                  left=0.06, right=0.94)


    input_img = sample_data[0].detach().cpu().numpy()
    input_img = input_img.transpose(1, 2, 0)  # (H, W, C)
    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)

    height, width = input_img.shape[0], input_img.shape[1]
    x_step = max(1, width // 5) 
    y_step = max(1, height // 5)
    x_ticks = np.arange(0, width, x_step)
    y_ticks = np.arange(0, height, y_step)


    cam = cam[0, 0]


    channels = ['HB Fingerprint', 'BS Fingerprint', 'SPB Fingerprint']
    for i in range(3):
        # ---- 原始通道显示 ----
        ax_orig = fig.add_subplot(gs[i, 0])
        ax_orig.imshow(input_img[..., i],cmap='gray',aspect='equal') #cmap='gray',

        # 设置统一刻度
        ax_orig.set_xticks(x_ticks)
        ax_orig.set_yticks(y_ticks)
        ax_orig.set_xticklabels([f"{int(x)}" for x in x_ticks], fontsize=8)
        ax_orig.set_yticklabels([f"{int(y)}" for y in y_ticks], fontsize=8)

        ax_orig.set_title(channels[i], fontsize=11, pad=10, y=0.95, fontweight='bold')

        # ---- 热力图叠加 ----
        ax_overlay = fig.add_subplot(gs[i, 1])
        ax_overlay.imshow(input_img[..., i],cmap='gray',aspect='equal')  # cmap='gray',
        overlay = ax_overlay.imshow(cam, cmap='jet', alpha=0.5, aspect='equal')

        # 设置相同刻度（隐藏标签）
        ax_overlay.set_xticks(x_ticks)
        ax_overlay.set_yticks(y_ticks)


        ax_overlay.set_title("Grad-CAM Overlay", fontsize=11, pad=10, y=0.95, fontweight='bold')

    # ---- Colorbar ----
    cax = fig.add_subplot(gs[:, 2])
    cbar = plt.colorbar(overlay, cax=cax, fraction=0.8)
    cbar.set_label('Activation Intensity', fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=9)

    plt.suptitle("Multi-Channel Grad-CAM Visualization", y=0.96, fontsize=14, fontweight='bold')
    plt.savefig("gradcam_result.png", dpi=300, bbox_inches='tight')
    plt.close()
