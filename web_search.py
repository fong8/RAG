"""
联网搜索模块 - 集成 Tavily API 进行网络搜索

使用前需在 .env 中配置 TAVILY_API_KEY
"""

import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def web_search(query, max_results=5):
    """
    使用 Tavily API 进行联网搜索。

    Args:
        query: 搜索关键词
        max_results: 最大返回结果数

    Returns:
        str: 格式化的搜索结果文本
    """
    if not TAVILY_API_KEY:
        return "错误：未配置 TAVILY_API_KEY，请在 .env 文件中添加。"

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=max_results)

        results = response.get("results", [])
        if not results:
            return "未找到相关的网络搜索结果。"

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            url = r.get("url", "")
            content = r.get("content", "无内容")
            formatted.append(f"[{i}] {title}\n链接: {url}\n摘要: {content}")

        return "\n\n---\n\n".join(formatted)

    except ImportError:
        return "错误：未安装 tavily-python，请运行 pip install tavily-python"
    except Exception as e:
        return f"搜索出错: {str(e)}"


# 工具定义（供 OpenAI function calling 使用）
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "当用户询问最新新闻、实时信息、本地文献库中没有的通用知识，或需要联网查询时，调用此工具进行网络搜索。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "网络搜索的关键词"}
            },
            "required": ["query"]
        }
    }
}
