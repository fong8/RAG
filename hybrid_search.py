"""
混合检索模块 - 融合向量检索 + BM25 关键词检索

通过 RRF (Reciprocal Rank Fusion) 合并两路检索结果，
再用 Cross-Encoder 重排，提升精准度。
"""
import hashlib
import re
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from reranker import rerank

# ---- BM25 索引缓存 ----
# 避免每次查询都全量加载文档并重建索引
_bm25_cache = {
   "collection_hash": "",       # 上次构建时的文档数量
    "bm25": None,          # BM25Okapi 实例
    "corpus_docs": [],     # 对应的 Document 列表
}

# BM25 最大索引文档数（超过此值随机采样，防止 OOM）
BM25_MAX_DOCS = 5000


def _tokenize(text: str) -> list[str]:
    """简单的中英文分词"""
    tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())
    return tokens

def _get_collection_hash(collection) -> str:
    """
    快速获取当前数据库状态的唯一指纹。
    只拉取文档 ID，不拉取几十兆的正文，速度极快且不占内存。
    """
    try:
        # include=[] 表示什么肉都不带，只要骨架（自动返回 ids）
        result = collection.get(include=[])
        if not result or not result.get("ids"):
            return ""
        
        # 将所有 ID 排序后拼成一个长字符串（保证顺序一致）
        ids_sorted = sorted(result["ids"])
        id_string = "".join(ids_sorted)
        
        # 计算并返回 MD5 哈希值（类似于 32 位的身份证号）
        return hashlib.md5(id_string.encode('utf-8')).hexdigest()
    except Exception:
        return ""

def _get_or_build_bm25(vector_db):
    """
    获取或重建 BM25 索引（带缓存）。

    只有当集合文档数量发生变化时才重建索引，
    否则直接复用缓存的 BM25 实例。

    Returns:
        (bm25_instance, doc_list) 或 (None, [])
    """
    global _bm25_cache

    try:
        collection = vector_db._collection
        current_count = collection.count()
    except Exception:
        return None, []

    if current_count == 0:
        return None, []
    current_hash = _get_collection_hash(collection)
    # 缓存命中：文档数量没变，直接复用
    if current_hash and current_hash == _bm25_cache["collection_hash"] and _bm25_cache["bm25"] is not None:
        return _bm25_cache["bm25"], _bm25_cache["corpus_docs"]
    # 缓存失效：重新构建索引
    try:
        # 分页加载文档，控制单次内存峰值
        all_docs = []
        batch_size = 1000
        offset = 0

        while offset < min(current_count, BM25_MAX_DOCS):
            batch = collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            if not batch or not batch["documents"]:
                break
            for content, meta in zip(batch["documents"], batch["metadatas"]):
                all_docs.append(Document(page_content=content, metadata=meta or {}))
            offset += batch_size

        if not all_docs:
            return None, []

        # 构建 BM25 索引
        corpus = [_tokenize(doc.page_content) for doc in all_docs]
        bm25 = BM25Okapi(corpus)

        # 写入缓存
        _bm25_cache["collection_hash"] = current_hash
        _bm25_cache["bm25"] = bm25
        _bm25_cache["corpus_docs"] = all_docs

        return bm25, all_docs

    except Exception:
        return None, []


def invalidate_bm25_cache():
    """手动清除 BM25 缓存（增删文档后调用）"""
    global _bm25_cache
    _bm25_cache = {"doc_count": -1, "bm25": None, "corpus_docs": []}


def _bm25_search(query: str, all_docs: list[Document], top_k: int = 10) -> list[Document]:
    """
    BM25 关键词检索（使用预构建的索引）

    Args:
        query: 查询文本
        all_docs: 全部文档列表
        top_k: 返回前 k 个结果
    """
    if not all_docs:
        return []

    # 如果传入的 all_docs 与缓存一致，直接用缓存的 bm25 实例
    bm25 = _bm25_cache.get("bm25")
    if bm25 is None or _bm25_cache["corpus_docs"] is not all_docs:
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

    # 2. BM25 检索 — 使用缓存索引，避免全量加载
    bm25, all_docs = _get_or_build_bm25(vector_db)
    if bm25 is not None and all_docs:
        bm25_results = _bm25_search(query, all_docs, top_k=bm25_candidates)
    else:
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
