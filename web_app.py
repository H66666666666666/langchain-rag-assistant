"""
LangChain RAG 文档问答助手 —— Web 界面
========================================
启动：  python web_app.py
访问：  http://127.0.0.1:5000

复用 app.py 中的 RAGAssistant，把 CLI 的全部能力搬到网页上：
  - 模型选择（DeepSeek / MiMo / OpenAI）+ API Key 配置
  - 加载 docs/ 下的 txt 文档、上传新文档、重建索引
  - 智能文本分割参数（chunk_size / chunk_overlap / search_k）
  - 基于文档的问答（带检索来源与相似度）
  - 向量检索（相似度评分）
  - 问答历史
"""
import os
from pathlib import Path

from flask import Flask, request, jsonify, render_template

from app import RAGAssistant, BASE_DIR

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

_PROVIDERS = ("deepseek", "mimo", "openai")
_KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "mimo": "MIMO_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# ---- 全局状态（单进程内存态，重启即清空历史）----
_state = {
    "model_provider": "deepseek",
    "api_key": None,
    "chunk_size": 500,
    "chunk_overlap": 50,
    "search_k": 3,
    "assistant": None,
    "initialized": False,
    "doc_count": 0,
    "chunk_count": 0,
    "history": [],
}


def _build_assistant():
    """根据当前配置构造（或重建）RAGAssistant 对象，不初始化索引。"""
    a = RAGAssistant(
        docs_dir="./docs",
        model_provider=_state["model_provider"],
        api_key=_state["api_key"],
    )
    a.chunk_size = _state["chunk_size"]
    a.chunk_overlap = _state["chunk_overlap"]
    a.search_k = _state["search_k"]
    _state["assistant"] = a
    _state["initialized"] = False
    _state["doc_count"] = 0
    _state["chunk_count"] = 0
    return a


def _ensure_initialized():
    if _state["assistant"] is None:
        _build_assistant()
    if not _state["initialized"]:
        raise RuntimeError("索引尚未初始化，请先点击「初始化 / 重建索引」。")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    provider = _state["model_provider"]
    has_key = bool(_state["api_key"] or os.getenv(_KEY_ENV[provider]))
    return jsonify({
        "model_provider": provider,
        "initialized": _state["initialized"],
        "doc_count": _state["doc_count"],
        "chunk_count": _state["chunk_count"],
        "has_key": has_key,
        "chunk_size": _state["chunk_size"],
        "chunk_overlap": _state["chunk_overlap"],
        "search_k": _state["search_k"],
    })


@app.route("/api/config", methods=["POST"])
def api_config():
    data = request.get_json(silent=True) or {}
    provider = data.get("model_provider", _state["model_provider"])
    if provider not in _PROVIDERS:
        return jsonify({"ok": False, "error": "不支持的模型提供商"}), 400

    _state["model_provider"] = provider
    if data.get("api_key"):
        # 写入环境变量，使 RAGAssistant 的回退逻辑也能取到
        os.environ[_KEY_ENV[provider]] = data["api_key"]
        _state["api_key"] = data["api_key"]
    try:
        _state["chunk_size"] = int(data.get("chunk_size", _state["chunk_size"]))
        _state["chunk_overlap"] = int(data.get("chunk_overlap", _state["chunk_overlap"]))
        _state["search_k"] = int(data.get("search_k", _state["search_k"]))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "参数必须为整数"}), 400

    # 配置变更后标记需重建索引
    _build_assistant()
    return jsonify({"ok": True})


@app.route("/api/init", methods=["POST"])
def api_init():
    try:
        a = _state["assistant"] or _build_assistant()
        a.initialize()
        _state["initialized"] = True
        _state["doc_count"] = getattr(a, "doc_count", 0)
        _state["chunk_count"] = getattr(a, "chunk_count", 0)
        return jsonify({
            "ok": True,
            "doc_count": _state["doc_count"],
            "chunk_count": _state["chunk_count"],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "问题不能为空"}), 400
    try:
        _ensure_initialized()
        a = _state["assistant"]
        answer, sources = a.ask_with_sources(question)
        _state["history"].append({
            "question": question,
            "answer": answer,
            "sources": sources,
        })
        return jsonify({"ok": True, "answer": answer, "sources": sources})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    k = int(data.get("k", _state["search_k"]))
    if not query:
        return jsonify({"ok": False, "error": "查询不能为空"}), 400
    try:
        _ensure_initialized()
        a = _state["assistant"]
        results = a.search(query, k=k)
        out = [{"content": doc.page_content, "similarity": sim} for doc, sim in results]
        return jsonify({"ok": True, "results": out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/docs", methods=["GET"])
def api_docs():
    docs_dir = BASE_DIR / "docs"
    files = sorted(p.name for p in docs_dir.glob("*.txt")) if docs_dir.exists() else []
    return jsonify({"ok": True, "files": files})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "未收到文件"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "文件名为空"}), 400
    if not f.filename.lower().endswith(".txt"):
        return jsonify({"ok": False, "error": "仅支持 .txt 文件"}), 400
    docs_dir = BASE_DIR / "docs"
    docs_dir.mkdir(exist_ok=True)
    save_path = docs_dir / f.filename
    f.save(str(save_path))
    return jsonify({"ok": True, "filename": f.filename})


@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify({"ok": True, "history": _state["history"]})


@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    _state["history"] = []
    return jsonify({"ok": True})


@app.route("/api/rebuild", methods=["POST"])
def api_rebuild():
    """上传文档后调用，等价于重新初始化索引。"""
    return api_init()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
