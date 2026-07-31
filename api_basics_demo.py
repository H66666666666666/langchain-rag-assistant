"""
api_basics_demo.py
不用任何 API Key、不联网，帮你直观看清：
"调用大模型 API" 到底在底层做了什么。

运行：python api_basics_demo.py
"""
import json
import os


# ---------------------------------------------------------------
# 第 1 部分：LangChain 帮你封装的，本质上就是下面这封信（HTTP 请求）
# ---------------------------------------------------------------
def show_raw_request():
    url = "https://api.deepseek.com/chat/completions"   # ① 餐厅地址（发给谁）
    api_key = os.getenv("DEEPSEEK_API_KEY", "<你的密钥>")  # ② 会员卡（证明你有资格）
    question = "什么是 RAG？"

    request = {
        "url": url,
        "headers": {
            "Authorization": f"Bearer {api_key}",        # 鉴权：证明是你
            "Content-Type": "application/json"
        },
        "body": {
            "model": "deepseek-chat",
            "temperature": 0,                             # 火候：0=别自由发挥
            "messages": [
                {"role": "user", "content": question}      # 你到底要模型做什么
            ]
        }
    }
    print("=== 一次 API 调用，本质上就是发出这样一封信 ===")
    print(json.dumps(request, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------
# 第 2 部分：模拟"服务器回信"，以及 LangChain 怎么把回信变成字符串
# ---------------------------------------------------------------
def show_response_parsing():
    # 这是服务器真实返回的 JSON（这里用假数据演示结构）
    fake_server_response = {
        "choices": [
            {"message": {"role": "assistant",
                         "content": "RAG 是检索增强生成，先检索资料再让模型回答。"}}
        ]
    }
    # LangChain 的 StrOutputParser() 干的事，其实就是下面这一行：
    answer = fake_server_response["choices"][0]["message"]["content"]
    print("\n=== 服务器回信（被 LangChain 解析后） ===")
    print(answer)


# ---------------------------------------------------------------
# 第 3 部分：你的项目里，LangChain 那一行等于上面全部
#   self.llm = ChatOpenAI(model="deepseek-chat", temperature=0,
#                         api_key=..., base_url="https://api.deepseek.com")
#   self.chain.invoke(question)   # <- 真正发信 + 收信
# ---------------------------------------------------------------
if __name__ == "__main__":
    show_raw_request()
    show_response_parsing()
    print("\n提示：你项目里的 self.chain.invoke(...) 一句话 = 发上面那封信 + 收上面那封回信。")
