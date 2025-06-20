# data/preprocess.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
def preprocess_ml_100k(
    data_path=r'D:\美琪的魔仙堡\代码文件\LLMRecProject\data\ml-100k\u.data',
    output_dir=r'D:\美琪的魔仙堡\代码文件\LLMRecProject\data\processed'
):
    os.makedirs(output_dir, exist_ok=True)
    # 读取数据
    df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df = df.drop(columns=['timestamp'])

    # 划分训练/测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    # 随机选择10个冷启动用户
    user_counts = train_df['user_id'].value_counts()
    cold_users = user_counts[user_counts <= 5].index.tolist()
    if len(cold_users) < 10:
        cold_users = train_df['user_id'].drop_duplicates().sample(10, random_state=42).tolist()

    # 对这些用户只保留一条记录（模拟冷启动）
    cold_user_data = train_df[train_df['user_id'].isin(cold_users)]
    keep_one = cold_user_data.groupby('user_id').head(1)
    train_df = train_df[~train_df['user_id'].isin(cold_users)]
    train_df = pd.concat([train_df, keep_one], ignore_index=True)
    train_df.to_csv(os.path.join(output_dir, 'train_cold.csv'), index=False)
    pd.DataFrame({'user_id': cold_users}).to_csv(os.path.join(output_dir, 'cold_users.csv'), index=False)
    print("数据预处理完成，已保存至:")
    print(f"  - 训练集: {output_dir}\\train.csv")
    print(f"  - 测试集: {output_dir}\\test.csv")
    print(f"  - 冷启动训练集: {output_dir}\\train_cold.csv")
    print(f"  - 冷启动用户列表: {output_dir}\\cold_users.csv")

if __name__ == "__main__":
    preprocess_ml_100k()
