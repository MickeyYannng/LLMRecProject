# utils/visualize_results.py

import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体和负号支持
plt.rcParams['font.family'] = 'SimHei'  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def score_interest_matching(user_tags, recommended_items):
    """
    改进版打分逻辑：
    用一个关键词映射字典，判断推荐电影名是否包含兴趣对应的关键词。
    如果包含，则该兴趣计数加1。
    """
    # 关键词映射：兴趣 -> 可能出现的关键词列表
    keyword_map = {
        "科幻": ["科幻", "星际", "太空", "未来", "机器人", "外星"],
        "篮球": ["篮球", "灌篮", "NBA", "扣篮", "篮球赛"],
        "太空": ["太空", "星际", "宇宙", "宇航"],
        "人工智能": ["人工智能", "AI", "机器人", "智能"],
        "青春": ["青春", "校园", "高中", "大学", "成长", "爱情"]
    }

    scores = {tag: 0 for tag in user_tags}

    for item in recommended_items:
        for tag in user_tags:
            keywords = keyword_map.get(tag, [tag])  # 如果没定义映射，默认用tag本身
            if any(kw in item for kw in keywords):
                scores[tag] += 1

    return scores

def draw_radar_chart(scores, save_path):
    if not scores:
        print("无法绘制雷达图：兴趣评分为空")
        return
    if all(v == 0 for v in scores.values()):
        print("所有评分为 0，跳过雷达图绘制")
        return

    labels = list(scores.keys())
    values = list(scores.values())
    num_vars = len(labels)

    print("🎯 雷达图兴趣评分：", scores)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="blue", linewidth=2)
    ax.fill(angles, values, color="skyblue", alpha=0.4)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("推荐兴趣覆盖雷达图", fontsize=14)
    ax.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"雷达图已保存到：{save_path}")

def draw_bar_chart(scores, save_path):
    if not scores:
        print("无法绘制柱状图：兴趣评分为空")
        return
    if all(v == 0 for v in scores.values()):
        print("所有评分为 0，跳过柱状图绘制")
        return

    labels = list(scores.keys())
    values = list(scores.values())

    print("柱状图兴趣评分：", scores)

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="orange")
    plt.title("推荐与兴趣匹配评分", fontsize=14)
    plt.xlabel("兴趣标签")
    plt.ylabel("匹配次数")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"柱状图已保存到：{save_path}")

# 示例调用
if __name__ == "__main__":
    user_tags = ["科幻", "篮球", "太空", "人工智能", "青春"]
    recommended_items = ["星际穿越", "灌篮高手", "头号玩家", "人工智能崛起", "校园时光"]

    scores = score_interest_matching(user_tags, recommended_items)
    draw_radar_chart(scores, "../output/chart_radar.png")
    draw_bar_chart(scores, "../output/chart_bar.png")

    print("图表生成完毕，输出位于 output 文件夹")
