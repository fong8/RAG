"""
数据可视化模块 - 生成图表并保存为图片

支持柱状图、折线图、饼图、散点图等，
通过 matplotlib 生成后保存为 PNG 供 Streamlit 展示。
"""

import os
import io
import json
import base64
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHART_DIR = "./charts"


def _ensure_chart_dir():
    os.makedirs(CHART_DIR, exist_ok=True)


def create_chart(
    chart_type: str,
    data: str,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> dict:
    """
    根据数据生成图表。

    Args:
        chart_type: 图表类型 (bar / line / pie / scatter)
        data: JSON 格式的数据，如：
            柱状图/折线图: {"labels": ["A","B"], "values": [10, 20]}
            多系列: {"labels": ["A","B"], "series": [{"name":"S1","values":[1,2]}, ...]}
            饼图: {"labels": ["A","B"], "values": [30, 70]}
            散点图: {"x": [1,2,3], "y": [4,5,6]}
        title: 图表标题
        x_label: X 轴标签
        y_label: Y 轴标签

    Returns:
        dict: {"path": 图片路径, "base64": base64编码}
    """
    _ensure_chart_dir()

    try:
        parsed = json.loads(data) if isinstance(data, str) else data
    except json.JSONDecodeError:
        return {"error": f"数据格式错误，需要 JSON 格式。收到: {data[:200]}"}

    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        chart_type = chart_type.lower().strip()

        if chart_type == "bar":
            _draw_bar(ax, parsed)
        elif chart_type == "line":
            _draw_line(ax, parsed)
        elif chart_type == "pie":
            _draw_pie(ax, parsed)
        elif chart_type == "scatter":
            _draw_scatter(ax, parsed)
        else:
            return {"error": f"不支持的图表类型: {chart_type}（支持: bar, line, pie, scatter）"}

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        plt.tight_layout()

        # 保存为文件
        filename = f"chart_{chart_type}_{hash(title) % 10000}.png"
        filepath = os.path.join(CHART_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")

        # 转 base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()

        plt.close(fig)

        return {"path": filepath, "filename": filename, "base64": b64}

    except Exception as e:
        plt.close(fig)
        return {"error": f"生成图表失败: {str(e)}"}


def _draw_bar(ax, data: dict):
    """绘制柱状图"""
    labels = data.get("labels", [])
    if "series" in data:
        import numpy as np
        n_series = len(data["series"])
        x = np.arange(len(labels))
        width = 0.8 / n_series
        for i, s in enumerate(data["series"]):
            offset = (i - n_series / 2 + 0.5) * width
            ax.bar(x + offset, s["values"], width, label=s.get("name", f"系列{i+1}"))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
    else:
        values = data.get("values", [])
        colors = plt.cm.Set2.colors[:len(labels)]
        ax.bar(labels, values, color=colors)


def _draw_line(ax, data: dict):
    """绘制折线图"""
    labels = data.get("labels", [])
    if "series" in data:
        for s in data["series"]:
            ax.plot(labels, s["values"], marker="o", label=s.get("name", ""))
        ax.legend()
    else:
        values = data.get("values", [])
        ax.plot(labels, values, marker="o", linewidth=2, color="#2196F3")
    ax.grid(True, alpha=0.3)


def _draw_pie(ax, data: dict):
    """绘制饼图"""
    labels = data.get("labels", [])
    values = data.get("values", [])
    colors = plt.cm.Set3.colors[:len(labels)]
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.axis("equal")


def _draw_scatter(ax, data: dict):
    """绘制散点图"""
    x = data.get("x", [])
    y = data.get("y", [])
    ax.scatter(x, y, alpha=0.7, s=60, color="#FF5722")
    ax.grid(True, alpha=0.3)


# 工具定义
CHART_TOOL = {
    "type": "function",
    "function": {
        "name": "create_chart",
        "description": "当用户需要数据可视化、生成图表、画图或对比数据时，调用此工具。支持柱状图(bar)、折线图(line)、饼图(pie)、散点图(scatter)。",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie", "scatter"],
                    "description": "图表类型"
                },
                "data": {
                    "type": "string",
                    "description": "JSON 格式数据。柱状图/折线图: {\"labels\":[\"A\",\"B\"],\"values\":[10,20]}; 多系列: {\"labels\":[...],\"series\":[{\"name\":\"S1\",\"values\":[...]}]}; 散点图: {\"x\":[1,2],\"y\":[3,4]}"
                },
                "title": {
                    "type": "string",
                    "description": "图表标题"
                },
                "x_label": {
                    "type": "string",
                    "description": "X轴标签"
                },
                "y_label": {
                    "type": "string",
                    "description": "Y轴标签"
                }
            },
            "required": ["chart_type", "data", "title"]
        }
    }
}
