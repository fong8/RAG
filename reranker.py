"""
重排器模块 - 使用 Cross-Encoder 对检索结果进行精排

初次使用时会自动下载模型（约 100MB），之后会使用缓存。
"""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# 使用轻量级中英文跨编码器模型
_DEFAULT_MODEL = "BAAI/bge-reranker-base"
_reranker = None


def _get_reranker():
    """懒加载重排器模型"""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_DEFAULT_MODEL)
    return _reranker


def rerank(query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
    """
    使用 Cross-Encoder 对候选文档进行重排序。

    Args:
        query: 用户查询
        documents: 候选文档列表 (LangChain Document)
        top_k: 返回排序后的前 k 个文档

    Returns:
        list[Document]: 按相关性从高到低排序的文档列表
    """
    if not documents:
        return []

    if len(documents) <= 1:
        return documents

    reranker = _get_reranker()

    # 构造 (query, doc_content) 对
    pairs = [(query, doc.page_content) for doc in documents]

    # 计算相关性分数
    scores = reranker.predict(pairs)

    # 按分数降序排列
    scored_docs = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in scored_docs[:top_k]]
