import os
import re
import json
import shutil
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from file_loader import load_file, SUPPORTED_EXTENSIONS
from query_rewriter import rewrite_query
from web_search import web_search, WEB_SEARCH_TOOL
from code_runner import run_python_code, CODE_RUNNER_TOOL
from hybrid_search import hybrid_search
from file_export import generate_file, list_exports, FILE_EXPORT_TOOL
from context_manager import truncate_messages
from visualizer import create_chart, CHART_TOOL

load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
DB_DIR = "./chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def process_uploaded_file(uploaded_file):
    """处理上传文件：解析文件 -> 切分 -> 追加存入向量库"""
    global vector_db
    documents = load_file(uploaded_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    vector_db.add_documents(chunks)
    return len(chunks)


def clear_vector_db():
    """清空整个向量数据库"""
    global vector_db
    if os.path.exists(DB_DIR):
        vector_db.delete_collection()
        shutil.rmtree(DB_DIR)
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def delete_file_from_db(file_name):
    """从向量数据库中删除指定文件的所有文档"""
    global vector_db
    collection = vector_db._collection
    results = collection.get(where={"source": file_name})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])
        return len(results["ids"])
    return 0


def get_indexed_files():
    """获取知识库中已索引的所有文件名及其片段数"""
    try:
        collection = vector_db._collection
        results = collection.get(include=["metadatas"])
        if not results or not results["metadatas"]:
            return {}
        file_counts = {}
        for meta in results["metadatas"]:
            source = meta.get("source", "未知文件")
            file_counts[source] = file_counts.get(source, 0) + 1
        return file_counts
    except Exception:
        return {}

st.title("RAG项目测试")
with st.sidebar:
    st.header("知识库管理")
    st.info(f"支持上传的文件格式：{', '.join('.' + e for e in SUPPORTED_EXTENSIONS)}")

    uploaded_file = st.file_uploader("请选择文件", type=SUPPORTED_EXTENSIONS)

    if uploaded_file is not None:
        if st.button("📥 添加到知识库"):
            with st.spinner("正在提取文本与计算向量，请稍候..."):
                try:
                    chunk_count = process_uploaded_file(uploaded_file)
                    st.success(f"处理完成。已将文档切分为 {chunk_count} 个片段存入知识库。")
                    st.rerun()
                except Exception as e:
                    st.error(f"处理文件时发生错误: {str(e)}")

    # --- 已索引文件列表 ---
    st.divider()
    st.subheader("📋 已索引文件")
    indexed_files = get_indexed_files()
    if indexed_files:
        for fname, count in indexed_files.items():
            col1, col2 = st.columns([3, 1])
            display_name = os.path.basename(fname) if "/" in fname or "\\" in fname else fname
            col1.caption(f"📄 {display_name}（{count} 片段）")
            if col2.button("🗑️", key=f"del_{fname}", help=f"删除 {display_name}"):
                deleted = delete_file_from_db(fname)
                st.success(f"已删除 {display_name}（{deleted} 个片段）")
                st.rerun()
    else:
        st.caption("知识库为空，请上传文件。")

    # 清空全部按钮
    if indexed_files:
        st.divider()
        if st.button("🗑️ 清空全部知识库", type="secondary"):
            clear_vector_db()
            st.success("知识库已清空。")
            st.rerun()

    # --- 已导出文件列表 ---
    exports = list_exports()
    if exports:
        st.divider()
        st.subheader("📦 已导出文件")
        for exp in exports:
            col1, col2 = st.columns([3, 1])
            col1.caption(f"📎 {exp['filename']}")
            with open(exp["path"], "rb") as ef:
                col2.download_button(
                    "⬇️", data=ef.read(),
                    file_name=exp["filename"],
                    key=f"sidebar_dl_{exp['filename']}"
                )

def search_local_papers(query, top_k=3):
    """混合检索：向量 + BM25 + RRF融合 + Cross-Encoder重排"""
    results = hybrid_search(query, vector_db, top_k=top_k)
    if not results:
        return "未找到相关文档。"
    formatted_docs = []
    for doc in results:
        source = doc.metadata.get('source', '未知文件')
        page = doc.metadata.get('page', '未知')
        content_type = doc.metadata.get('content_type', 'text')
        if isinstance(page, int):
            page += 1
        content = doc.page_content
        type_tag = ""
        if content_type == "table":
            type_tag = " [表格]"
        elif content_type == "image":
            type_tag = " [图片]"
        doc_string = f"[文献出处: {source}, 第 {page} 页{type_tag}]\n{content}"
        formatted_docs.append(doc_string)
    return "\n\n---\n\n".join(formatted_docs)
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_local_papers",
            "description": "当用户询问关于实验室、特定论文、本地文献或需要查阅专业学术资料时，调用此工具检索信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用于搜索的关键词"}
                },
                "required": ["query"]
            }
        }
    },
    WEB_SEARCH_TOOL,
    CODE_RUNNER_TOOL,
    FILE_EXPORT_TOOL,
    CHART_TOOL,
]

# 工具函数注册表：名称 -> (执行函数, 图标, 显示名)
TOOL_REGISTRY = {
    "search_local_papers": (None, "📚", "本地文献检索"),
    "web_search": (web_search, "🌐", "联网搜索"),
    "run_python_code": (run_python_code, "💻", "Python 代码执行"),
    "generate_file": (generate_file, "📄", "文件生成"),
    "create_chart": (create_chart, "📊", "数据可视化"),
}
client= OpenAI(api_key=deepseek_api_key,base_url="https://api.deepseek.com/v1")
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system", 
        "content": (
            "你是一个多功能 AI 助手，拥有以下工具：\n"
            "1. **search_local_papers** - 检索本地知识库中的文献\n"
            "2. **web_search** - 联网搜索最新信息\n"
            "3. **run_python_code** - 执行 Python 代码进行数学计算或数据分析\n"
            "4. **generate_file** - 生成并导出文件（Markdown/TXT/CSV）\n"
            "5. **create_chart** - 生成数据可视化图表（柱状图/折线图/饼图/散点图）\n\n"
            "当用户询问关于'我的论文'、'实验'或具体知识时，你**必须**调用 search_local_papers 工具去检索。\n"
            "当用户需要最新信息、新闻或本地文献库没有的内容时，使用 web_search。\n"
            "当用户需要数学计算、数据分析或代码执行时，使用 run_python_code。\n"
            "当用户要求生成报告、总结或导出数据时，使用 generate_file。\n"
            "当用户需要图表、可视化或数据对比时，使用 create_chart。\n\n"
            "【核心要求】\n"
            "在回答用户问题时，如果你使用了工具检索到的内容，**必须**在引用的句子或段落末尾标注来源！\n"
            "本地文献标注格式：(参考: 文件名, 第X页)。网络搜索标注格式：(来源: 链接)。"
        )
    }]
for msg in st.session_state.messages:
    if msg["role"] in ("user", "assistant"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
if prompt := st.chat_input("请输入你的问题"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # 动态上下文截断：发送前压缩消息历史
            messages_to_send = truncate_messages(st.session_state.messages, max_tokens=12000)
            response = client.chat.completions.create(model="deepseek-chat",messages=messages_to_send,tools=tools,tool_choice="auto")
            response_message = response.choices[0].message
            content_text = response_message.content or ""
            
            # --- 多工具分发逻辑 ---
            tool_called = False
            
            if response_message.tool_calls:
                tool_called = True
                # 将助手消息（含 tool_calls）加入历史
                st.session_state.messages.append({
                    "role": response_message.role,
                    "content": content_text,
                    "tool_calls": [tc.model_dump() for tc in response_message.tool_calls]
                })
                
                # 逐个执行工具调用
                for tool_call in response_message.tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    registry_entry = TOOL_REGISTRY.get(fn_name)
                    
                    if not registry_entry:
                        tool_result = f"未知工具: {fn_name}"
                    else:
                        fn, icon, display_name = registry_entry
                        
                        if fn_name == "search_local_papers":
                            # 本地检索：会话感知查询重写
                            search_query = args.get("query", "")
                            rewritten = rewrite_query(client, st.session_state.messages, search_query)
                            if rewritten != search_query:
                                st.info(f"{icon} {display_name}: `{search_query}` → `{rewritten}`")
                            else:
                                st.info(f"{icon} {display_name}: `{search_query}`")
                            tool_result = search_local_papers(rewritten)
                        
                        elif fn_name == "web_search":
                            query = args.get("query", "")
                            st.info(f"{icon} {display_name}: `{query}`")
                            tool_result = web_search(query)
                        
                        elif fn_name == "run_python_code":
                            code = args.get("code", "")
                            st.info(f"{icon} {display_name}")
                            with st.expander("查看执行的代码", expanded=False):
                                st.code(code, language="python")
                            tool_result = run_python_code(code)
                        
                        elif fn_name == "generate_file":
                            fname_arg = args.get("filename", "export")
                            content_arg = args.get("content", "")
                            ftype = args.get("file_type", "md")
                            st.info(f"{icon} {display_name}: `{fname_arg}.{ftype}`")
                            result = generate_file(fname_arg, content_arg, ftype)
                            if "path" in result:
                                # 提供下载按钮
                                with open(result["path"], "rb") as f:
                                    st.download_button(
                                        label=f"⬇️ 下载 {result['filename']}",
                                        data=f.read(),
                                        file_name=result["filename"],
                                        key=f"dl_{result['filename']}_{id(tool_call)}"
                                    )
                            tool_result = json.dumps(result, ensure_ascii=False)
                        
                        elif fn_name == "create_chart":
                            ctype = args.get("chart_type", "bar")
                            cdata = args.get("data", "{}")
                            ctitle = args.get("title", "")
                            cxlabel = args.get("x_label", "")
                            cylabel = args.get("y_label", "")
                            st.info(f"{icon} {display_name}: {ctitle}")
                            result = create_chart(ctype, cdata, ctitle, cxlabel, cylabel)
                            if "path" in result:
                                st.image(result["path"], caption=ctitle)
                            elif "error" in result:
                                st.error(result["error"])
                            tool_result = json.dumps(
                                {k: v for k, v in result.items() if k != "base64"},
                                ensure_ascii=False
                            )
                        
                        else:
                            tool_result = fn(**args) if fn else f"工具 {fn_name} 未实现"
                    
                    st.session_state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)
                    })
                
                # 所有工具执行完毕，LLM 生成最终回答
                messages_for_final = truncate_messages(st.session_state.messages, max_tokens=12000)
                final_response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages_for_final,
                    stream=True
                )
                full_response = ""
                for chunk in final_response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # --- 非标准工具调用回退（DSML 格式） ---
            if not tool_called:
                # 检查是否有非标准格式的工具调用
                match = re.search(r'parameter name="query".*?>(.*?)</', content_text)
                if match:
                    search_query = match.group(1)
                    rewritten = rewrite_query(client, st.session_state.messages, search_query)
                    if rewritten != search_query:
                        st.info(f"📚 本地文献检索: `{search_query}` → `{rewritten}`")
                    else:
                        st.info(f"📚 本地文献检索: `{search_query}`")
                    tool_result = search_local_papers(rewritten)
                    
                    clean_content = re.sub(r'<\|DSML\|function_calls>.*?</\|DSML\|function_calls>', '', content_text, flags=re.DOTALL).strip()
                    if clean_content:
                        st.session_state.messages.append({"role": "assistant", "content": clean_content})
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"以下是系统自动检索到的本地文献资料，请根据这些内容回答用户的问题：\n{tool_result}"
                    })
                    final_response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=st.session_state.messages,
                        stream=True
                    )
                    full_response = ""
                    for chunk in final_response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    clean_msg = re.sub(r'<\|DSML\|function_calls>.*?</\|DSML\|function_calls>', '', content_text, flags=re.DOTALL).strip()
                    message_placeholder.markdown(clean_msg)
                    st.session_state.messages.append({"role": "assistant", "content": clean_msg})
                
        except Exception as e:
            message_placeholder.markdown(f"发生错误: {str(e)}")