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

load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
DB_DIR = "./chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def process_uploaded_file(uploaded_file):
    """处理上传文件：清空旧数据 -> 解析文件 -> 切分 -> 存入向量库"""
    global vector_db
    if os.path.exists(DB_DIR):
        vector_db.delete_collection()
        shutil.rmtree(DB_DIR)
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    documents = load_file(uploaded_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    vector_db.add_documents(chunks)
    return len(chunks)

st.title("RAG项目测试")
with st.sidebar:
    st.header("知识库管理")
    st.info(f"支持上传的文件格式：{', '.join('.' + e for e in SUPPORTED_EXTENSIONS)}")

    uploaded_file = st.file_uploader("请选择文件", type=SUPPORTED_EXTENSIONS)

    if uploaded_file is not None:
        if st.button("开始处理并存入知识库"):
            with st.spinner("正在提取文本与计算向量，请稍候..."):
                try:
                    chunk_count = process_uploaded_file(uploaded_file)
                    st.success(f"处理完成。已将文档切分为 {chunk_count} 个片段存入知识库。")
                except Exception as e:
                    st.error(f"处理文件时发生错误: {str(e)}")

def search_local_papers(query,top_k=3):
    results = vector_db.similarity_search(query, k=top_k)
    if not results:
        return "未找到相关文档。"
    formatted_docs = []
    for doc in results:
        source = doc.metadata.get('source', '未知文件')
        page = doc.metadata.get('page', '未知') 
        if isinstance(page, int):
            page += 1       
        content = doc.page_content
        doc_string = f"[文献出处: {source}, 第 {page} 页]\n{content}"
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
]

# 工具函数注册表：名称 -> (执行函数, 图标, 显示名)
TOOL_REGISTRY = {
    "search_local_papers": (None, "📚", "本地文献检索"),  # 特殊处理，需要查询重写
    "web_search": (web_search, "🌐", "联网搜索"),
    "run_python_code": (run_python_code, "💻", "Python 代码执行"),
}
client= OpenAI(api_key=deepseek_api_key,base_url="https://api.deepseek.com/v1")
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system", 
        "content": (
            "你是一个多功能 AI 助手，拥有以下工具：\n"
            "1. **search_local_papers** - 检索本地知识库中的文献\n"
            "2. **web_search** - 联网搜索最新信息\n"
            "3. **run_python_code** - 执行 Python 代码进行数学计算或数据分析\n\n"
            "当用户询问关于'我的论文'、'实验'或具体知识时，你**必须**调用 search_local_papers 工具去检索。\n"
            "当用户需要最新信息、新闻或本地文献库没有的内容时，使用 web_search。\n"
            "当用户需要数学计算、数据分析或代码执行时，使用 run_python_code。\n\n"
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
            response = client.chat.completions.create(model="deepseek-chat",messages=st.session_state.messages,tools=tools,tool_choice="auto")
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
                        
                        else:
                            tool_result = fn(**args) if fn else f"工具 {fn_name} 未实现"
                    
                    st.session_state.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)
                    })
                
                # 所有工具执行完毕，LLM 生成最终回答
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