"""
文件加载模块 - 支持 PDF、Word (.docx)、TXT、PPT (.pptx) 文件解析
统一接口：load_file(uploaded_file) -> list[Document]
"""

import os
import tempfile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from docx import Document as DocxDocument
from pptx import Presentation


def _save_temp_file(uploaded_file, suffix):
    """将上传文件保存为临时文件，返回路径"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def _load_pdf(uploaded_file) -> list[Document]:
    """加载 PDF 文件"""
    tmp_path = _save_temp_file(uploaded_file, ".pdf")
    try:
        loader = PyMuPDFLoader(tmp_path)
        return loader.load()
    finally:
        os.remove(tmp_path)


def _load_docx(uploaded_file) -> list[Document]:
    """加载 Word (.docx) 文件"""
    tmp_path = _save_temp_file(uploaded_file, ".docx")
    try:
        doc = DocxDocument(tmp_path)
        documents = []
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name, "paragraph": i + 1}
                ))
        # 如果段落过多且零散，合并为按页（每10段一组）的文档
        if len(documents) > 20:
            merged = []
            for start in range(0, len(documents), 10):
                batch = documents[start:start + 10]
                content = "\n".join(d.page_content for d in batch)
                merged.append(Document(
                    page_content=content,
                    metadata={"source": uploaded_file.name, "page": start // 10 + 1}
                ))
            return merged
        return documents
    finally:
        os.remove(tmp_path)


def _load_txt(uploaded_file) -> list[Document]:
    """加载 TXT 文件"""
    raw = uploaded_file.getvalue()
    # 尝试多种编码
    for encoding in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = raw.decode("utf-8", errors="ignore")

    return [Document(
        page_content=text,
        metadata={"source": uploaded_file.name, "page": 1}
    )]


def _load_pptx(uploaded_file) -> list[Document]:
    """加载 PPT (.pptx) 文件"""
    tmp_path = _save_temp_file(uploaded_file, ".pptx")
    try:
        prs = Presentation(tmp_path)
        documents = []
        for slide_num, slide in enumerate(prs.slides, start=1):
            texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        t = paragraph.text.strip()
                        if t:
                            texts.append(t)
            if texts:
                documents.append(Document(
                    page_content="\n".join(texts),
                    metadata={"source": uploaded_file.name, "page": slide_num}
                ))
        return documents
    finally:
        os.remove(tmp_path)


# 格式 -> 加载函数 映射表
_LOADERS = {
    "pdf": _load_pdf,
    "docx": _load_docx,
    "txt": _load_txt,
    "pptx": _load_pptx,
}

# 支持的文件扩展名列表（供 Streamlit file_uploader 使用）
SUPPORTED_EXTENSIONS = list(_LOADERS.keys())


def load_file(uploaded_file) -> list[Document]:
    """
    统一文件加载接口。
    根据文件扩展名自动选择对应的加载器，返回 LangChain Document 列表。

    Args:
        uploaded_file: Streamlit UploadedFile 对象

    Returns:
        list[Document]: 解析后的文档列表

    Raises:
        ValueError: 不支持的文件格式
    """
    file_name = uploaded_file.name
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    loader_fn = _LOADERS.get(ext)
    if loader_fn is None:
        raise ValueError(f"不支持的文件格式: .{ext}（支持: {', '.join(SUPPORTED_EXTENSIONS)}）")

    return loader_fn(uploaded_file)
