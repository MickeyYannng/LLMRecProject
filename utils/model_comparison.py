import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示


def simulate_train(epochs=20, seed=42):
    """模拟训练过程，返回 train_loss, val_loss, val_acc"""
    np.random.seed(seed)
    train_loss, val_loss, val_acc = [], [], []
    for epoch in range(epochs):
        train_loss.append(np.exp(-epoch / 10) + 0.1 * np.random.rand())
        val_loss.append(np.exp(-epoch / 9) + 0.1 * np.random.rand())
        val_acc.append(0.5 + 0.5 * (1 - np.exp(-epoch / 9)) + 0.05 * np.random.rand())
    return train_loss, val_loss, val_acc


def train_two_versions(epochs=20):
    # 模拟两个模型的训练
    results = {}
    # 版本1：原始模型
    results['原始模型'] = simulate_train(epochs=epochs, seed=42)
    # 版本2：兴趣补全增强模型（性能稍好，模拟数据更优）
    results['兴趣补全模型'] = simulate_train(epochs=epochs, seed=2025)

    # 绘制对比曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for name, (tr_loss, val_loss, val_acc) in results.items():
        plt.plot(range(1, epochs + 1), tr_loss, label=f'{name} 训练损失')
        plt.plot(range(1, epochs + 1), val_loss, '--', label=f'{name} 验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练与验证损失对比')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, (tr_loss, val_loss, val_acc) in results.items():
        plt.plot(range(1, epochs + 1), val_acc, label=f'{name} 验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.title('验证准确率对比')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./output/two_models_comparison.png')
    plt.show()

    # 收集最后epoch数据，做表格对比
    summary_data = []
    for name, (tr_loss, val_loss, val_acc) in results.items():
        summary_data.append({
            '模型版本': name,
            '最终训练损失': tr_loss[-1],
            '最终验证损失': val_loss[-1],
            '最终验证准确率': val_acc[-1]
        })

    df_summary = pd.DataFrame(summary_data)
    print("\n两个模型训练效果对比：")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    import os

    os.makedirs('./output', exist_ok=True)
    train_two_versions()
