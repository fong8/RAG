# 📚 RAG 多功能知识库助手

一个基于 **检索增强生成（RAG）** 技术的智能问答系统，支持本地文档知识库检索、联网搜索、Python 代码执行、数据可视化与文件导出，前端由 Streamlit 驱动，大模型后端接入 DeepSeek。

---

## ✨ 核心功能

| 功能 | 描述 |
|------|------|
| 📄 **多格式文档解析** | 支持 PDF、Word(.docx)、TXT、PPT(.pptx)、Markdown(.md) |
| 🔍 **混合检索** | 向量检索 + BM25 关键词检索，通过 RRF 融合后 Cross-Encoder 重排 |
| 🔄 **会话感知查询重写** | 多轮对话中自动消除指代词，提升检索准确性 |
| 🌐 **联网搜索** | 集成 Tavily API，查询本地库以外的实时信息 |
| 💻 **Python 代码执行** | 沙箱环境内安全执行数学计算与数据分析代码 |
| 📊 **数据可视化** | 支持柱状图、折线图、饼图、散点图，结果直接展示在对话中 |
| 📦 **文件导出** | 生成 Markdown / TXT / CSV 文件，支持侧边栏一键下载 |
| 🧠 **动态上下文管理** | 智能截断对话历史，保持在 Token 上限内 |

---

## 🗂️ 项目结构

```
├── app.py                # 主应用入口，Streamlit UI + 多工具分发逻辑
├── file_loader.py        # 文件解析模块（PDF/DOCX/TXT/PPTX/MD）
├── hybrid_search.py      # 混合检索：向量 + BM25 + RRF 融合
├── reranker.py           # Cross-Encoder 重排器（BAAI/bge-reranker-base）
├── query_rewriter.py     # 会话感知查询重写
├── web_search.py         # Tavily 联网搜索
├── code_runner.py        # Python 沙箱代码执行
├── visualizer.py         # Matplotlib 图表生成
├── file_export.py        # Markdown/TXT/CSV 文件生成与导出
├── context_manager.py    # 动态对话上下文截断
├── .env                  # 环境变量（需自行创建）
├── chroma_db/            # Chroma 向量数据库持久化目录（自动生成）
├── exports/              # 导出文件存放目录（自动生成）
└── charts/               # 图表图片存放目录（自动生成）
```

---

## 🚀 快速开始

### 1. 环境依赖

**Python 版本要求**：3.10+

安装依赖：

```bash
pip install streamlit openai langchain langchain-community \
    chromadb sentence-transformers rank_bm25 \
    pymupdf pdfplumber python-docx python-pptx \
    matplotlib tavily-python python-dotenv
```

### 2. 配置 Ollama 嵌入模型

本项目使用 [Ollama](https://ollama.com/) 在本地运行嵌入模型：

```bash
# 安装 Ollama 后拉取嵌入模型
ollama pull nomic-embed-text
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here   # 可选，联网搜索功能需要
```

- DeepSeek API Key：前往 [platform.deepseek.com](https://platform.deepseek.com) 获取
- Tavily API Key：前往 [tavily.com](https://tavily.com) 获取（免费额度可用）

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可使用。

---

## 📖 使用说明

### 上传文档

在左侧侧边栏点击 **"请选择文件"** 上传文档，支持 `.pdf`、`.docx`、`.txt`、`.pptx`、`.md` 格式。上传后点击 **"📥 添加到知识库"**，系统将自动完成文本解析、分块（chunk_size=1000，overlap=150）和向量化存储。

### 知识库管理

- 侧边栏展示所有已索引文件及其分块数量
- 点击文件旁的 **🗑️** 可单独删除该文件的索引
- 点击 **"🗑️ 清空全部知识库"** 可重置整个向量数据库

### 对话与工具调用

直接在对话框输入问题，AI 会根据意图自动选择工具：

| 意图示例 | 触发工具 |
|---------|---------|
| "帮我查一下论文里关于…的内容" | 📚 本地文献检索 |
| "搜索一下最近关于…的新闻" | 🌐 联网搜索 |
| "计算一下这组数据的方差" | 💻 Python 代码执行 |
| "把这份总结导出为 Markdown 文件" | 📄 文件生成 |
| "用柱状图展示这些数据" | 📊 数据可视化 |

### 导出文件下载

AI 生成的文件会出现在对话中的下载按钮，同时也会在侧边栏 **"📦 已导出文件"** 区域保留，方便随时下载。

---

## 🏗️ 技术架构

```
用户输入
   │
   ▼
┌─────────────────────────────────────┐
│         查询重写（query_rewriter）    │  ← 多轮对话消歧
└───────────────┬─────────────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
  向量检索（Chroma）   BM25 关键词检索
  (Ollama Embeddings)  (rank_bm25)
        │                │
        └───────┬────────┘
                ▼
          RRF 融合排序
                │
                ▼
     Cross-Encoder 精排（重排器）
     (BAAI/bge-reranker-base)
                │
                ▼
        DeepSeek LLM 生成回答
```

---

## ⚙️ 关键配置参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `chunk_size` | `app.py` | 1000 | 文档分块大小（字符数） |
| `chunk_overlap` | `app.py` | 150 | 分块重叠大小 |
| `top_k` | `hybrid_search.py` | 3 | 最终返回文档数 |
| `vector_candidates` | `hybrid_search.py` | 10 | 向量检索候选数 |
| `bm25_candidates` | `hybrid_search.py` | 10 | BM25 检索候选数 |
| `max_tokens` | `context_manager.py` | 12000 | 上下文 Token 上限 |
| `reserve_recent` | `context_manager.py` | 6 | 始终保留最近消息数 |

---

## ⚠️ 注意事项

- **首次使用重排器**：`reranker.py` 会自动下载 `BAAI/bge-reranker-base` 模型（约 100MB），请确保网络畅通
- **代码执行安全**：沙箱禁止 `os`、`sys`、`subprocess` 等危险模块，仅允许数学计算和数据处理操作
- **PDF 增强解析**：同时使用 PyMuPDF（文本）、pdfplumber（表格）和 PyMuPDF（图片信息）三路解析，需确保相关库均已安装
- **中文字体**：图表中文显示依赖系统字体（Arial Unicode MS / SimHei），Linux 服务器部署时可能需要额外安装中文字体包

---

## 📄 License

MIT License
