from openai import OpenAI
client= OpenAI(api_key="sk-06d14b93876d47659d104860e74cd34b",base_url="https://api.deepseek.com/v1")
response = client.chat.completions.create(model="deepseek-chat",messages=[{"role": "system", "content": "你是一个严谨的AI方向学术导师。"},
        {"role": "user", "content": "用一句话解释什么是RAG（检索增强生成）？"}])
print(response.choices[0].message.content)