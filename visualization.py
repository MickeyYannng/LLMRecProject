# visualization.py

import matplotlib.pyplot as plt
import os

def plot_loss_curves(train_losses, val_losses, save_path="../outputs/loss_curve.png"):
    """
    绘制训练损失和验证损失曲线，并保存为图片。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', linestyle='--', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('训练损失 vs 验证损失', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"训练与验证损失图已保存至: {save_path}")

def plot_training_metrics(train_losses, val_losses):
    """
    主调用接口，用于自动绘图所有训练过程可视化图（目前支持图2）
    """
    plot_loss_curves(train_losses, val_losses)
