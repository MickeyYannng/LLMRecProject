# main.py

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils.llm_client import LLMClient
from utils.train_recommender import RatingDataset, MatrixFactorization, train_model

# 数据加载函数
def load_movielens_data():
    num_users, num_items = 1000, 1700
    np.random.seed(42)
    user_ids = np.random.randint(0, num_users, 5000)
    item_ids = np.random.randint(0, num_items, 5000)
    ratings = np.random.randint(0, 2, 5000)

    train_num = int(0.8 * len(user_ids))
    train_data = (user_ids[:train_num], item_ids[:train_num], ratings[:train_num])
    val_data = (user_ids[train_num:], item_ids[train_num:], ratings[train_num:])
    return train_data, val_data, num_users, num_items

# 模拟用户已有兴趣标签
def get_user_interests(num_users):
    # 简单生成一些兴趣标签字典示例
    interests = {}
    for uid in range(num_users):
        interests[uid] = ["电影", "篮球"] if uid % 2 == 0 else ["音乐", "旅游"]
    return interests

# 用 LLM 补全用户兴趣标签
def augment_user_interests(user_ids, user_interests, llm_client):
    augmented_interests = {}
    for user_id in user_ids:
        existing = user_interests.get(user_id, [])
        prompt = f"用户ID:{user_id}，现有兴趣:{','.join(existing)}。请补全更多兴趣标签，提供5条。"
        result = llm_client.generate(prompt)
        if result:
            interests = [i.strip() for i in result.split(',') if i.strip()]
            augmented_interests[user_id] = interests
        else:
            augmented_interests[user_id] = existing
    return augmented_interests

def main():
    print("项目启动...")

    # 1. 初始化LLM客户端
    api_key = "36789a53ef7e55d41fdcd8ccd967cb6c.eZDJWT4PSdftJbrH"
    llm_client = LLMClient(api_key)

    # 2. 加载数据
    train_data, val_data, num_users, num_items = load_movielens_data()
    user_interests = get_user_interests(num_users)

    # 3. 数据增强：LLM补全用户兴趣
    user_ids = list(range(num_users))
    augmented_interests = augment_user_interests(user_ids, user_interests, llm_client)
    print("用户兴趣补全示例：")
    for uid in list(augmented_interests.keys())[:3]:
        print(f"用户 {uid} 兴趣标签: {augmented_interests[uid]}")

    # 4. 构造数据集与训练
    train_dataset = RatingDataset(*train_data)
    val_dataset = RatingDataset(*val_data)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    model = MatrixFactorization(num_users, num_items, emb_size=32)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    trained_model = train_model(train_loader, val_loader, model, optimizer, criterion, epochs=20)

    print("项目执行完成，训练结果和图表已保存。")

if __name__ == "__main__":
    main()
