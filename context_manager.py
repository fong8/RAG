"""
动态上下文截断模块 - 智能管理对话历史的 token 长度

通过估算 token 数量，在超出限制时智能截断早期消息，
保留 system 提示、最近对话和工具调用上下文。
"""


def estimate_tokens(text: str) -> int:
    """
    估算文本 token 数量。
    中文约 1 字 = 1.5 token，英文约 1 词 = 1.3 token。
    """
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    # 英文按空格/标点切词估算
    return int(chinese_chars * 1.5 + other_chars * 0.4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """估算整个消息列表的 token 数"""
    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        total += estimate_tokens(content) + 4  # role + formatting overhead
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                total += estimate_tokens(fn.get("name", ""))
                total += estimate_tokens(fn.get("arguments", ""))
    return total


def truncate_messages(
    messages: list[dict],
    max_tokens: int = 12000,
    reserve_recent: int = 6,
) -> list[dict]:
    """
    智能截断对话历史，控制在 token 上限内。

    策略：
    1. 始终保留 system 消息
    2. 始终保留最近 N 轮对话
    3. 中间的 tool 消息优先压缩（截短 content）
    4. 如仍超限，丢弃最早的 user/assistant 消息

    Args:
        messages: 完整消息列表
        max_tokens: 最大 token 限制
        reserve_recent: 保留最近的消息数（不含 system）

    Returns:
        list[dict]: 截断后的消息列表
    """
    if not messages:
        return messages

    # 分离 system 消息和其他
    system_msgs = [m for m in messages if m["role"] == "system"
                   and messages.index(m) == 0]  # 只保留首条 system
    other_msgs = [m for m in messages if m not in system_msgs]

    current_tokens = estimate_messages_tokens(messages)
    if current_tokens <= max_tokens:
        return messages

    # --- 第一步：压缩 tool 消息中过长的 content ---
    tool_content_limit = 1500  # 每条工具结果最多保留的字符数
    compressed = list(messages)  # 浅拷贝
    for i, msg in enumerate(compressed):
        if msg["role"] == "tool" and len(msg.get("content", "")) > tool_content_limit:
            compressed[i] = {
                **msg,
                "content": msg["content"][:tool_content_limit] + "\n...(内容已截断)"
            }

    if estimate_messages_tokens(compressed) <= max_tokens:
        return compressed

    # --- 第二步：保留 system + 最近 N 条，丢弃中间消息 ---
    if len(other_msgs) <= reserve_recent:
        return compressed

    recent = other_msgs[-reserve_recent:]
    # 确保不会切断 tool_call 对：如果 recent 开头是 tool 消息，向前扩展
    while recent and recent[0]["role"] == "tool" and len(other_msgs) > len(recent):
        idx = other_msgs.index(recent[0])
        if idx > 0:
            recent.insert(0, other_msgs[idx - 1])
        else:
            break

    result = system_msgs + recent

    # --- 第三步：如果仍超限，进一步压缩 ---
    while estimate_messages_tokens(result) > max_tokens and len(result) > len(system_msgs) + 2:
        # 删除 system 后的第一条非 system 消息
        for i in range(len(system_msgs), len(result)):
            if result[i]["role"] != "system":
                result.pop(i)
                break

    return result
