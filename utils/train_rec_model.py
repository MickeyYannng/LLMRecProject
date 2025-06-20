# utils/train_rec_model.py (示例框架)
import numpy as np
import matplotlib.pyplot as plt
# 设置中文字体和负号支持
plt.rcParams['font.family'] = 'SimHei'  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
def train_model():
    epochs = 20
    train_loss = []
    val_loss = []
    val_acc = []

    for epoch in range(epochs):
        # 训练过程
        train_loss.append(np.exp(-epoch / 10) + 0.1 * np.random.rand())
        val_loss.append(np.exp(-epoch / 9) + 0.1 * np.random.rand())
        val_acc.append(0.5 + 0.5 * (1 - np.exp(-epoch / 9)) + 0.05 * np.random.rand())

        print(f"Epoch {epoch+1}/{epochs}, train_loss={train_loss[-1]:.4f}, val_loss={val_loss[-1]:.4f}, val_acc={val_acc[-1]:.4f}")

    # 绘制训练和验证损失折线图
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs+1), train_loss, label='训练损失')
    plt.plot(range(1, epochs+1), val_loss, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失和验证损失')
    plt.legend()
    plt.savefig('../output/train_val_loss.png')
    plt.close()

    # 绘制验证准确率折线图
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs+1), val_acc, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('验证准确率')
    plt.legend()
    plt.savefig('../output/val_accuracy.png')
    plt.close()

    print("训练曲线图已保存")

if __name__ == "__main__":
    train_model()
