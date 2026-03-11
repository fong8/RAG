"""
文件生成与导出模块 - 生成 Markdown / TXT / CSV 文件供用户下载

LLM 可通过工具调用生成结构化内容并导出为文件。
"""

import os
import csv
import io

EXPORT_DIR = "./exports"


def _ensure_export_dir():
    """确保导出目录存在"""
    os.makedirs(EXPORT_DIR, exist_ok=True)


def generate_file(filename: str, content: str, file_type: str = "md") -> dict:
    """
    生成文件并保存到导出目录。

    Args:
        filename: 文件名（不含扩展名）
        content: 文件内容
        file_type: 文件类型 (md / txt / csv)

    Returns:
        dict: {"path": 文件路径, "size": 文件大小}
    """
    _ensure_export_dir()

    # 清理文件名
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ").strip()
    if not safe_name:
        safe_name = "export"

    ext = file_type.lower()
    if ext not in ("md", "txt", "csv"):
        ext = "md"

    filepath = os.path.join(EXPORT_DIR, f"{safe_name}.{ext}")

    if ext == "csv":
        # CSV: 按行解析，支持 | 分隔的表格格式
        rows = _parse_table_to_rows(content)
        if rows:
            with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    size = os.path.getsize(filepath)
    return {"path": filepath, "size": size, "filename": f"{safe_name}.{ext}"}


def _parse_table_to_rows(text: str) -> list[list[str]]:
    """尝试将 Markdown 表格或文本解析为 CSV 行"""
    lines = text.strip().split("\n")
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 跳过 Markdown 表格分隔行 (|---|---|)
        if set(line.replace("|", "").replace("-", "").replace(" ", "")) == set():
            continue
        if "|" in line:
            cells = [cell.strip() for cell in line.split("|")]
            # 去掉首尾空元素
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]
            rows.append(cells)
        elif "," in line:
            rows.append([cell.strip() for cell in line.split(",")])
        else:
            rows.append([line])
    return rows


def list_exports() -> list[dict]:
    """列出所有已导出的文件"""
    _ensure_export_dir()
    files = []
    for f in os.listdir(EXPORT_DIR):
        path = os.path.join(EXPORT_DIR, f)
        if os.path.isfile(path):
            files.append({
                "filename": f,
                "path": path,
                "size": os.path.getsize(path),
            })
    return files


# 工具定义
FILE_EXPORT_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_file",
        "description": "当用户要求生成报告、总结、导出数据或创建文档时，调用此工具生成文件。支持 Markdown(.md)、文本(.txt)和表格(.csv)格式。",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "文件名（不含扩展名），如 'SOH综述报告'"
                },
                "content": {
                    "type": "string",
                    "description": "文件的完整内容。对于CSV可使用 | 分隔的表格或逗号分隔。"
                },
                "file_type": {
                    "type": "string",
                    "enum": ["md", "txt", "csv"],
                    "description": "文件格式：md(Markdown报告)、txt(纯文本)、csv(表格数据)"
                }
            },
            "required": ["filename", "content", "file_type"]
        }
    }
}
