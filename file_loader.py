"""
文件加载模块 - 支持 PDF、Word (.docx)、TXT、PPT (.pptx)、Markdown (.md) 文件解析
统一接口：load_file(uploaded_file) -> list[Document]
"""
import os
import tempfile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from docx import Document as DocxDocument
from pptx import Presentation
import re

def _save_temp_file(uploaded_file, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def _load_pdf(uploaded_file) -> list[Document]:
    tmp_path = _save_temp_file(uploaded_file, ".pdf")
    try:
        loader = PyMuPDFLoader(tmp_path)
        return loader.load()
    finally:
        os.remove(tmp_path)


def _load_docx(uploaded_file) -> list[Document]:
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


def _load_markdown(uploaded_file) -> list[Document]:
    """加载 Markdown (.md) 文件，按标题分段"""
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = raw.decode("utf-8", errors="ignore")

    sections = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
    documents = []
    for i, section in enumerate(sections):
        content = section.strip()
        if content:
            documents.append(Document(
                page_content=content,
                metadata={"source": uploaded_file.name, "page": i + 1}
            ))

    if not documents:
        documents.append(Document(
            page_content=text,
            metadata={"source": uploaded_file.name, "page": 1}
        ))
    return documents


_LOADERS = {
    "pdf": _load_pdf,
    "docx": _load_docx,
    "txt": _load_txt,
    "pptx": _load_pptx,
    "md": _load_markdown,
}

SUPPORTED_EXTENSIONS = list(_LOADERS.keys())


def load_file(uploaded_file) -> list[Document]:
    file_name = uploaded_file.name
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    loader_fn = _LOADERS.get(ext)
    if loader_fn is None:
        raise ValueError(f"不支持的文件格式: .{ext}（支持: {', '.join(SUPPORTED_EXTENSIONS)}）")

    return loader_fn(uploaded_file)
