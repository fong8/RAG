"""
混合检索模块 - 融合向量检索 + BM25 关键词检索

通过 RRF (Reciprocal Rank Fusion) 合并两路检索结果，
再用 Cross-Encoder 重排，提升精准度。
"""

import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from reranker import rerank


def _tokenize(text: str) -> list[str]:
    """简单的中英文分词"""
    # 中文按字符分，英文按空格/标点分
    tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())
    return tokens


def _bm25_search(query: str, all_docs: list[Document], top_k: int = 10) -> list[Document]:
    """
    BM25 关键词检索

    Args:
        query: 查询文本
        all_docs: 全部文档列表
        top_k: 返回前 k 个结果
    """
    if not all_docs:
        return []

    corpus = [_tokenize(doc.page_content) for doc in all_docs]
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    scored_docs = sorted(
        zip(scores, all_docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in scored_docs[:top_k]]


def _rrf_merge(
    vector_results: list[Document],
    bm25_results: list[Document],
    k: int = 60
) -> list[Document]:
    """
    Reciprocal Rank Fusion 合并两路检索结果。

    RRF score = sum(1 / (k + rank_i)) for each retriever

    Args:
        vector_results: 向量检索结果（已排序）
        bm25_results: BM25 检索结果（已排序）
        k: RRF 常数（默认 60）

    Returns:
        合并后按 RRF 分数降序排列的文档列表
    """
    doc_scores = {}  # doc_id -> (score, Document)

    for rank, doc in enumerate(vector_results):
        doc_id = id(doc)
        score = 1.0 / (k + rank + 1)
        doc_scores[doc_id] = (doc_scores.get(doc_id, (0, doc))[0] + score, doc)

    # BM25 结果可能与向量结果有重叠，用 page_content 去重
    content_to_id = {id(doc): doc.page_content for doc in vector_results}
    existing_contents = set(content_to_id.values())

    for rank, doc in enumerate(bm25_results):
        score = 1.0 / (k + rank + 1)
        # 检查内容是否已存在（去重）
        matched_id = None
        for did, content in content_to_id.items():
            if content == doc.page_content:
                matched_id = did
                break

        if matched_id:
            old_score, old_doc = doc_scores[matched_id]
            doc_scores[matched_id] = (old_score + score, old_doc)
        else:
            doc_id = id(doc)
            doc_scores[doc_id] = (score, doc)
            content_to_id[doc_id] = doc.page_content

    # 按 RRF 分数降序排列
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs]


def hybrid_search(
    query: str,
    vector_db,
    top_k: int = 3,
    vector_candidates: int = 10,
    bm25_candidates: int = 10,
    use_rerank: bool = True,
) -> list[Document]:
    """
    混合检索：向量检索 + BM25 关键词检索 + RRF 融合 + 重排。

    Args:
        query: 查询文本
        vector_db: Chroma 向量数据库实例
        top_k: 最终返回的文档数
        vector_candidates: 向量检索候选数量
        bm25_candidates: BM25 检索候选数量
        use_rerank: 是否使用 Cross-Encoder 重排

    Returns:
        list[Document]: 检索结果
    """
    # 1. 向量检索
    vector_results = vector_db.similarity_search(query, k=vector_candidates)

    # 2. BM25 检索 — 从 Chroma 取全部文档
    try:
        collection = vector_db._collection
        all_data = collection.get(include=["documents", "metadatas"])
        all_docs = []
        if all_data and all_data["documents"]:
            for content, meta in zip(all_data["documents"], all_data["metadatas"]):
                all_docs.append(Document(page_content=content, metadata=meta or {}))
        bm25_results = _bm25_search(query, all_docs, top_k=bm25_candidates)
    except Exception:
        bm25_results = []

    # 3. RRF 融合
    if bm25_results:
        merged = _rrf_merge(vector_results, bm25_results)
    else:
        merged = vector_results

    if not merged:
        return []

    # 4. Cross-Encoder 重排
    if use_rerank and len(merged) > 1:
        return rerank(query, merged, top_k=top_k)

    return merged[:top_k]
