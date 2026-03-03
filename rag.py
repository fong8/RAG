from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import numpy as np
import requests
docs = [
    "公司成立于2020年，总部位于北京。",
    "主要产品是AI助手和数据分析平台。",
    "客服电话是 400-123-4567，工作时间9:00-18:00。",
    "退款政策：7天无理由退款，需保持商品完好。",
]
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
embeddings = model.encode(docs)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))
def retrieve(query,top_k=2):
    query_vec = model.encode([query])
    _,indices = index.search(np.array(query_vec),top_k)
    return [docs[i] for i in indices[0]]

client = OpenAI(base_url="http://127.0.0.1:11434/v1",api_key="ollama")
def rag(query):
    context = "\n".join(retrieve(query))
    prompt = f"""根据以下资料回答问题，不要编造内容。

资料：
{context}

问题：{query}
"""
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": "qwen2.5:latest", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]
print(rag("客服电话是多少？"))
print(rag("你们什么时候成立的？"))