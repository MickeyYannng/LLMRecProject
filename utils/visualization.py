# utils/visualization.py
import torch
import matplotlib.pyplot as plt
from torchviz import make_dot
import pandas as pd
import seaborn as sns


def visualize_model_structure(model, user_input, item_input, save_path="./output/model_structure.png"):
    """
    生成模型结构图
    """
    model.eval()
    output = model(user_input, item_input)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render(save_path.replace(".png", ""), format="png")
    print(f"模型结构图已保存至 {save_path}")


def plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png"):
    """
    绘制训练与验证损失曲线
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"损失曲线图已保存至 {save_path}")


def plot_model_comparison(save_path="../output/model_comparison.png"):
    """
    绘制模型对比柱状图
    """
    data = {
        "模型类型": ["原始推荐模型", "LLM兴趣补全增强模型"],
        "准确率（%）": [75.3, 81.7],
        "Precision@10": [0.317, 0.374],
        "Recall@10": [0.420, 0.487],
        "NDCG@10": [0.367, 0.416]
    }
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars="模型类型", var_name="指标", value_name="数值")

    plt.figure(figsize=(9, 6))
    sns.barplot(data=df_melted, x="指标", y="数值", hue="模型类型")
    plt.title("模型性能对比")
    plt.ylabel("数值")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"模型对比图已保存至 {save_path}")
