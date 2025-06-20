# utils/augment_interest.py

import pandas as pd
from llm_client import LLMClient

def load_movie_titles(item_path='../data/ml-100k/u.item'):
    df = pd.read_csv(item_path, sep='|', encoding='latin-1', header=None)
    df.columns = ['item_id', 'title'] + [f'col_{i}' for i in range(22)]
    return df[['item_id', 'title']]

def build_prompt(user_id, df_user_ratings, movie_titles):
    movies = df_user_ratings['item_id'].map(movie_titles.set_index('item_id')['title'])
    movie_list = '、'.join(movies.tolist())
    return f"用户对以下电影有评分：{movie_list}。请根据这些电影补全用户兴趣标签，提供5条简洁的兴趣描述，用顿号分隔。"

def augment_user_interest(cold_user_file, train_file, item_file, output_file, api_key):
    # 加载数据
    cold_users = pd.read_csv(cold_user_file)
    train_df = pd.read_csv(train_file)
    movie_titles = load_movie_titles(item_file)

    # 初始化LLM客户端
    client = LLMClient(api_key)

    results = []

    for uid in cold_users['user_id']:
        user_ratings = train_df[train_df['user_id'] == uid]
        prompt = build_prompt(uid, user_ratings, movie_titles)
        print(f"[User {uid}] Prompt => {prompt}")

        response = client.generate(prompt)
        print(f"[User {uid}] Response => {response}")

        results.append({
            'user_id': uid,
            'prompt': prompt,
            'augmented_interest': response
        })

    pd.DataFrame(results).to_csv(output_file, index=False)
    print("补全完成，结果保存至:", output_file)

if __name__ == "__main__":
    augment_user_interest(
        cold_user_file='../data/processed/cold_users.csv',
        train_file='../data/processed/train_cold.csv',
        item_file='../data/ml-100k/u.item',
        output_file='../data/processed/user_interest_augmented.csv',
        api_key="36789a53ef7e55d41fdcd8ccd967cb6c.eZDJWT4PSdftJbrH"
    )
