# utils/llm_client.py

import requests

class LLMClient:
    def __init__(self, api_key, model="glm-4"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def generate(self, prompt, max_tokens=2048, temperature=0.7):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print("请求失败:", e)
            return None
        except (KeyError, IndexError) as e:
            print("返回格式错误:", e)
            print("返回内容:", response.text)
            return None

# 测试调用
if __name__ == "__main__":
    api_key = "36789a53ef7ebrH"
    client = LLMClient(api_key=api_key)

    prompt = "用户兴趣：喜欢科幻电影和篮球。请补全用户兴趣标签，提供5条。"
    result = client.generate(prompt)
    print("LLM生成结果:", result)
