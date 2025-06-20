# utils/train_recommender.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from torchsummary import summary

# 简单矩阵分解模型
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, emb_size=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_emb(user_ids)
        item_vecs = self.item_emb(item_ids)
        return (user_vecs * item_vecs).sum(dim=1)

# 自定义数据集
class RatingDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

def train_model(train_loader, val_loader, model, optimizer, criterion, epochs=20, device='cpu'):
    model.to(device)
    train_losses, val_losses = [], []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            optimizer.zero_grad()
            preds = model(u, i)
            loss = criterion(preds, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(r)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for u, i, r in val_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                preds = model(u, i)
                loss = criterion(preds, r)
                total_val_loss += loss.item() * len(r)

                # 简单准确率：预测评分>0.5为正，标签>0.5为正
                pred_labels = (preds > 0.5).float()
                true_labels = (r > 0.5).float()
                correct += (pred_labels == true_labels).sum().item()
                total += len(r)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        val_acc = correct / total
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}")

    # 绘图
    os.makedirs('./output', exist_ok=True)
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='训练损失')
    plt.plot(range(1, epochs+1), val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.savefig('./output/train_val_loss.png')
    plt.close()

    plt.figure()
    plt.plot(range(1, epochs+1), val_accuracies, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.title('验证准确率曲线')
    plt.legend()
    plt.savefig('./output/val_accuracy.png')
    plt.close()

    print("训练曲线图已保存至 ./output 文件夹")

    return model

# 加载数据示例
def load_movielens_data():
    num_users, num_items = 1000, 1700
    np.random.seed(42)
    user_ids = np.random.randint(0, num_users, 5000)
    item_ids = np.random.randint(0, num_items, 5000)
    ratings = np.random.randint(0, 2, 5000)  # 0/1 评分简化版

    # 划分训练验证
    train_num = int(0.8 * len(user_ids))
    train_data = (user_ids[:train_num], item_ids[:train_num], ratings[:train_num])
    val_data = (user_ids[train_num:], item_ids[train_num:], ratings[train_num:])
    return train_data, val_data, num_users, num_items

if __name__ == "__main__":
    train_data, val_data, num_users, num_items = load_movielens_data()

    train_dataset = RatingDataset(*train_data)
    val_dataset = RatingDataset(*val_data)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    model = MatrixFactorization(num_users, num_items, emb_size=32)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    trained_model = train_model(train_loader, val_loader, model, optimizer, criterion, epochs=20)


    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型可训练参数总数: {total_params:,}")


    print("模型架构：")
    print(model)  # 假设模型对象为model
    count_parameters(model)


    def print_model_params_table(model):
        print("层名称\t\t参数数量\t\t参数形状")
        print("-" * 40)
        for name, param in model.named_parameters():
            print(f"{name}\t{param.numel():,}\t{tuple(param.shape)}")


    print_model_params_table(model)
