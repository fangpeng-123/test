from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import requests
# 配置SerpApi Key
SERPAPI_KEY = "d33b4179de64636c6325a1fb136299c505cded7a3dcd43d3609c89d551457565"

def google_search(query):
    """
    使用 SerpAPI 进行 Google 搜索，并返回前5个结果的标题和摘要。
    """
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 5
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("orangic_results", [])
        snippets = ""
        for res in results:
            title = res.get("title", "")
            snippet = res.get("snippets", "")
            snippets += f"【{title}】\n{snippet}\n\n"
        return snippets
    else:
        return "无法联网搜索结果，请检查网络或者 API Key。"
    
# 初始化大模型
ollm = OllamaLLM(model="qwen3:8b")

# 定义提示模板
Prompt_Template = """
你是一个知识问答助手，请根据以下提供的网络搜索结果来回答问题。

【网络搜索结果】
{context}

【用户问题】
{question}

请基于以上信息，用中文简洁明了地回答。
"""

prompt = PromptTemplate.from_template(Prompt_Template)

# 构建LLM Chain
from langchain_core.runnables import RunnableSequence
chain = prompt | ollm

# 查询函数
def answer_question(question):
    context = google_search(question)
    response  =chain.invoke({"context": context, "question": question})
    return response

# 测试
if __name__ == "__main__":
    question = input("请问你想知道什么：")
    print("🔍 正在联网搜索中...")
    answer = answer_question(question)
    print("\n📝 回答：")
    print(answer)