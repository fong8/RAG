"""
会话感知查询重写模块

在多轮对话中，用户的后续问题往往包含指代词（如"它"、"这个"、"上面提到的"），
直接用这些模糊语句检索会导致召回质量差。

本模块通过 LLM 根据对话历史将当前问题重写为一个独立、完整的检索查询词，
从而提升向量检索的准确性。
"""


def rewrite_query(client, messages, current_query, model="deepseek-chat"):
    """
    根据对话历史重写用户查询，使其成为一个独立且适合检索的查询语句。

    Args:
        client: OpenAI 客户端实例
        messages: 当前对话历史 (list[dict])
        current_query: 用户当前输入的原始问题 (str)
        model: 使用的模型名称

    Returns:
        str: 重写后的查询词
    """
    # 提取最近的对话上下文（最多取最近 6 轮 user/assistant 消息）
    recent_context = []
    for msg in messages:
        if msg["role"] in ("user", "assistant"):
            recent_context.append(msg)
    recent_context = recent_context[-6:]

    # 如果没有历史对话，无需重写
    if not recent_context:
        return current_query

    # 构造对话摘要
    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else 'AI'}: {m['content']}"
        for m in recent_context
    )

    rewrite_messages = [
        {
            "role": "system",
            "content": (
                "你是一个查询重写助手。你的任务是根据对话历史，将用户最新的问题重写为一个独立、完整、适合用于文献检索的查询语句。\n\n"
                "规则：\n"
                "1. 消除所有指代词（如'它'、'这个'、'该方法'），替换为具体名词\n"
                "2. 保留用户的核心意图，不要添加多余内容\n"
                "3. 只输出重写后的查询语句，不要输出任何解释\n"
                "4. 如果问题已经足够清晰完整，直接原样返回"
            )
        },
        {
            "role": "user",
            "content": f"对话历史：\n{history_text}\n\n用户最新问题：{current_query}\n\n请重写为独立的检索查询语句："
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=rewrite_messages,
        temperature=0,
        max_tokens=150,
    )

    rewritten = response.choices[0].message.content.strip()
    # 去除可能的引号包裹
    if (rewritten.startswith('"') and rewritten.endswith('"')) or \
       (rewritten.startswith("'") and rewritten.endswith("'")):
        rewritten = rewritten[1:-1]

    return rewritten if rewritten else current_query
