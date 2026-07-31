# LangChain RAG 文档问答助手

一个使用 LangChain 实现的文档问答系统，可以加载文档并基于文档内容回答问题。

## 功能特性

- 📄 支持加载 txt 文档
- ✂️ 智能文本分割
- 🔍 向量检索
- 💬 基于文档的问答
- 📊 相似度评分

## 项目结构

```
langchain-rag-assistant/
├── app.py              # 主程序
├── requirements.txt    # 依赖
├── docs/               # 文档目录
│   └── sample.txt      # 示例文档
└── README.md           # 说明文档
```

## 安装步骤

### 1. 克隆项目
```bash
cd langchain-rag-assistant
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 设置 API Key
```bash
# 方法1: 环境变量
export OPENAI_API_KEY=your-key

# 方法2: 创建 .env 文件
echo "OPENAI_API_KEY=your-key" > .env
```

## 使用方法

### 运行程序
```bash
python app.py
```

### 交互示例
```
💬 开始提问（输入 'quit' 退出）

你：什么是机器学习？
🤔 思考中...
🤖 回答：机器学习是人工智能的一个子领域，它使计算机能够从数据中学习，而无需显式编程。

你：RAG的主要步骤有哪些？
🤔 思考中...
🤖 回答：RAG的主要步骤包括：1. 文档加载；2. 文本分割；3. 向量化；4. 存储；5. 检索；6. 生成。
```

## 核心组件

| 组件 | 作用 |
|------|------|
| TextLoader | 加载文本文档 |
| RecursiveCharacterTextSplitter | 智能分割文本 |
| OpenAIEmbeddings / HuggingFaceEmbeddings | 文本向量化（OpenAI 用前者，deepseek/mimo 用本地 HuggingFace 模型） |
| Chroma | 向量存储和检索（使用 cosine 距离） |
| ChatOpenAI | 语言模型 |

> 注：`assistant.search(query)` 返回的是「相似度」而非距离，已做 `相似度 = 1 - 距离` 换算（cosine 距离，范围约 -1~1，越大越相关）。若想拿到原始距离，可在 `search()` 中自行使用 `similarity_search_with_score`。

## 自定义配置

在 `app.py` 中修改 `RAGAssistant` 类的参数：

```python
assistant = RAGAssistant(
    docs_dir="./docs",      # 文档目录
    api_key="your-key"      # API Key
)

# 可选配置
assistant.chunk_size = 500      # 文本块大小
assistant.chunk_overlap = 50    # 重叠字符数
assistant.search_k = 3          # 检索结果数量
```

## 添加更多文档

将 txt 文件放入 `docs/` 目录，程序会自动加载。

## 学习要点

通过这个项目，你可以学到：

1. **文档加载** — 如何加载文档（本最小化示例仅支持 `.txt` 纯文本，见 `_load_documents`）
2. **文本分割** — 如何将长文档分割成小块
3. **向量化** — 如何将文本转换成向量
4. **向量存储** — 如何使用向量数据库
5. **检索增强生成** — 如何实现 RAG

## 扩展建议

- 支持更多文档格式（PDF、Word、网页）
- 添加对话记忆功能
- 使用本地模型替代 OpenAI
- 添加 Web 界面
- 优化检索策略

## 常见问题

### Q: 没有 OpenAI API Key 怎么办？
A: 可以使用本地模型，修改 `app.py` 中的模型配置。

### Q: 如何支持中文？
A: 当前已支持中文，确保文档是 UTF-8 编码。

### Q: 检索结果不准确？
A: 调整 `chunk_size` 和 `search_k` 参数。

## 依赖

- LangChain
- OpenAI
- ChromaDB
- Tiktoken

## 许可证

MIT License
