# utils/integrated_recommend_visualize.py

from llm_client import LLMClient
from visualize_results import score_interest_matching, draw_radar_chart, draw_bar_chart

def parse_recommended_movies(text):
    """
    简单解析推荐电影列表，假设是逗号分隔或换行分隔
    返回列表
    """
    # 先用逗号分割
    if "," in text:
        items = [x.strip() for x in text.split(",") if x.strip()]
    else:
        # 否则按换行拆分
        items = [x.strip() for x in text.split("\n") if x.strip()]
    return items

def main():
    api_key = "36789a53rH"
    client = LLMClient(api_key=api_key)

    # 用户兴趣标签（真实项目中可动态获得）
    user_tags = ["科幻", "篮球", "太空", "人工智能", "青春"]

    # prompt示例：让大模型生成推荐电影列表
    prompt = (
        "请根据用户兴趣标签：科幻、篮球、太空、人工智能、青春，"
        "推荐10部相关的电影名称，"
        "只列出电影名称，使用逗号分隔。"
    )

    print("正在调用LLM获取推荐电影...")
    result_text = client.generate(prompt)

    if not result_text:
        print("未获取到推荐电影结果，退出")
        return

    print("LLM返回的推荐电影列表文本：", result_text)

    recommended_items = parse_recommended_movies(result_text)
    print("解析后的推荐电影列表:", recommended_items)

    scores = score_interest_matching(user_tags, recommended_items)

    # 生成图表保存路径
    radar_path = "../output/chart_radar.png"
    bar_path = "../output/chart_bar.png"

    draw_radar_chart(scores, radar_path)
    draw_bar_chart(scores, bar_path)

    print("基于API调用结果的图表生成完毕")

if __name__ == "__main__":
    main()
