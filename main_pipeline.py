from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

# ===== 加载并训练模型 =====
file_path = r'D:\美琪的魔仙堡\代码文件\LLMRecProject\data\ml-100k\u.data'
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

# ===== 电影ID到名称映射 =====
item_id_to_name = {}
with open(r"D:\美琪的魔仙堡\代码文件\LLMRecProject\data\ml-100k\u.item", encoding="ISO-8859-1") as f:
    for line in f:
        parts = line.strip().split("|")
        item_id_to_name[parts[0]] = parts[1]


# ===== 封装函数：获取指定用户的推荐电影 =====
def get_recommendations_for_user(user_id, top_n=5):
    # 获取用户未评分的所有电影
    all_items = trainset.all_items()
    raw_item_ids = [trainset.to_raw_iid(iid) for iid in all_items]

    rated_items = set([j for (j, _) in trainset.ur[trainset.to_inner_uid(user_id)]])
    unrated_items = [iid for iid in raw_item_ids if trainset.to_inner_iid(iid) not in rated_items]

    # 为未评分电影预测评分
    predictions = []
    for iid in unrated_items:
        pred = model.predict(user_id, iid)
        predictions.append((iid, pred.est))

    # 按评分排序，选出 top-N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_items = [item_id_to_name.get(iid, f"电影ID:{iid}") for iid, _ in predictions[:top_n]]

    return top_items


#  测试
if __name__ == "__main__":
    test_user_id = "196"
    recs = get_recommendations_for_user(test_user_id, top_n=5)
    print(f"推荐给用户 {test_user_id} 的电影：")
    for idx, movie in enumerate(recs, 1):
        print(f"{idx}. {movie}")
