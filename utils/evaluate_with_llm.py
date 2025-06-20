from llm_client import LLMClient

API_KEY = "36789a53ef7e5"

def generate_evaluation(user_tags, recommended_items):
    prompt = (
        f"用户兴趣标签：{user_tags}\n"
        f"推荐电影：{recommended_items}\n"
        "请判断推荐电影是否符合用户兴趣，并用一句话简要说明理由。"
    )

    client = LLMClient(api_key=API_KEY)
    response = client.generate(prompt)
    return response

# 示例运行
if __name__ == "__main__":
    user_tags = ["科幻", "篮球", "太空冒险"]
    recommended_items = ["星际穿越", "灌篮高手", "头号玩家"]
    result = generate_evaluation(user_tags, recommended_items)
    print("评估语句:", result)
