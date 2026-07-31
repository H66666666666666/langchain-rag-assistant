"""
LangChain RAG 文档问答助手
功能：加载文档 → 建立索引 → 智能问答
支持：DeepSeek / Mimo / OpenAI
"""

import os
import shutil
import sys
from pathlib import Path
from dotenv import load_dotenv

# LangChain 组件
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 加载环境变量
load_dotenv()

# 脚本所在目录，用于锚定相对路径，避免依赖当前工作目录 (CWD)
BASE_DIR = Path(__file__).parent


# ========== 命令行颜色工具（ANSI 转义码） ==========
class Color:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def c(text: str, color: str) -> str:
    """给文本上色；非交互式终端（管道/重定向）时不加 ANSI 码，避免污染日志"""
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{Color.RESET}"


class RAGAssistant:
    """RAG 文档问答助手"""

    def __init__(self, docs_dir: str = "./docs", model_provider: str = "deepseek", api_key: str = None):
        """
        初始化助手

        Args:
            docs_dir: 文档目录路径
            model_provider: 模型提供商 ("deepseek", "mimo", "openai")
                注：ChatOpenAI 通过 base_url 路由到对应服务；
                其中 deepseek / mimo 使用本地 HuggingFace 向量模型做 embedding，
                仅 openai 使用 OpenAIEmbeddings。
            api_key: API Key
        """
        self.docs_dir = BASE_DIR / docs_dir
        self.model_provider = model_provider
        self.api_key = api_key

        # 初始化组件
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.chain = None

        # 统计信息（供 Web 界面展示）
        self.doc_count = 0
        self.chunk_count = 0

        # 设置
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.search_k = 3

    def initialize(self):
        """初始化所有组件"""
        print("🚀 正在初始化...")

        # 1. 初始化模型
        print(f"   加载语言模型 ({self.model_provider})...")
        if self.model_provider == "deepseek":
            self.llm = ChatOpenAI(
                model="deepseek-chat",
                temperature=0,
                api_key=self.api_key or os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
            # DeepSeek 不提供 OpenAI 兼容的 embeddings 端点，改用本地 HuggingFace 向量模型
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        elif self.model_provider == "mimo":
            self.llm = ChatOpenAI(
                model="mimo-chat",
                temperature=0,
                api_key=self.api_key or os.getenv("MIMO_API_KEY"),
                base_url="https://api.mimo.xiaomi.com"
            )
            # MiMo 同样不提供 OpenAI 兼容的 embeddings 端点，改用本地 HuggingFace 向量模型
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        elif self.model_provider == "openai":
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                api_key=self.api_key or os.getenv("OPENAI_API_KEY")
            )
            self.embeddings = OpenAIEmbeddings(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"不支持的模型提供商: {self.model_provider}")

        # 2. 加载文档
        print("   加载文档...")
        documents = self._load_documents()
        self.doc_count = len(documents)

        # 3. 分割文本
        print("   分割文本...")
        chunks = self._split_documents(documents)
        self.chunk_count = len(chunks)

        # 4. 创建向量库
        print("   创建向量库...")
        self._create_vectorstore(chunks)

        # 5. 创建问答链
        print("   创建问答链...")
        self._create_chain()

        print("✅ 初始化完成！\n")

    def _load_documents(self) -> list:
        """加载文档"""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"文档目录不存在: {self.docs_dir}")

        # 加载所有 txt 文件
        # silent_errors=True: 跳过编码异常/损坏的单个文件，避免整个程序崩溃
        loader = DirectoryLoader(
            str(self.docs_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True
        )

        documents = loader.load()
        print(f"   加载了 {len(documents)} 个文档")
        if not documents:
            print(c("⚠️ 警告：docs/ 下没有加载到任何 txt（可能目录为空，或文件编码非 UTF-8 被跳过）",
                    Color.YELLOW))
        return documents

    def _split_documents(self, documents: list) -> list:
        """分割文档"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " "]
        )

        chunks = splitter.split_documents(documents)
        print(f"   分割成 {len(chunks)} 个文本块")
        return chunks

    def _create_vectorstore(self, chunks: list):
        """创建向量库"""
        chroma_dir = BASE_DIR / "chroma_db"
        # 每次运行都重建索引：Chroma.from_documents 遇已存在的 collection 会"追加"而非重建，
        # 会导致旧文档重复、或在切换 embedding 模型后向量空间错乱。
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)

        # 使用 Chroma 向量库（统一用 cosine 距离，使相似度换算合理）
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(chroma_dir),
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"   向量库创建完成（{len(chunks)} 个文本块）")

    def _create_chain(self):
        """创建问答链"""
        # 创建检索器
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.search_k}
        )

        # 创建提示词模板
        prompt = ChatPromptTemplate.from_template("""
基于以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文：
{context}

问题：{question}

回答：""")

        # 创建问答链
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """
        提问

        Args:
            question: 问题

        Returns:
            回答
        """
        if not self.chain:
            raise RuntimeError("请先调用 initialize() 初始化")

        print(c(f"❓ 问题：{question}", Color.CYAN))
        print(c("🤔 思考中...", Color.YELLOW))

        # 调用问答链
        answer = self.chain.invoke(question)

        return answer

    def ask_with_sources(self, question: str, k: int = None) -> tuple:
        """
        提问并返回答案 + 检索到的来源片段（带相似度）

        Args:
            question: 问题
            k: 检索来源数量（默认用 self.search_k）

        Returns:
            (answer, sources) 元组；sources 为 [(文本, 相似度), ...]
        """
        if not self.chain:
            raise RuntimeError("请先调用 initialize() 初始化")

        answer = self.chain.invoke(question)
        k = k or self.search_k
        results = self.vectorstore.similarity_search_with_score(question, k=k)
        # Chroma 返回距离，转换为相似度（向量库使用 cosine 距离）
        sources = [(doc.page_content, 1 - score) for doc, score in results]
        return answer, sources

    def search(self, query: str, k: int = 3) -> list:
        """
        搜索相关文档

        Args:
            query: 查询
            k: 返回结果数量

        Returns:
            相关文档列表，元素为 (document, similarity) 元组；
            similarity = 1 - distance（cosine 距离，范围约 -1~1，越大越相关）
        """
        if not self.vectorstore:
            raise RuntimeError("请先调用 initialize() 初始化")

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # Chroma 返回的是"距离"（越低越相似），这里转换为"相似度"返回
        return [(doc, 1 - score) for doc, score in results]


def main():
    """主函数"""
    print("=" * 50)
    print("📚 LangChain RAG 文档问答助手")
    print("=" * 50)
    print()

    # 选择模型
    print("选择模型提供商：")
    print("1. DeepSeek（推荐，便宜）")
    print("2. Mimo（小米）")
    print("3. OpenAI")
    print()

    # 模型选择：循环校验，避免误输入静默落到 OpenAI
    while True:
        choice = input("请选择 (1/2/3，默认1): ").strip()
        if choice == "":
            choice = "1"
        if choice in ("1", "2", "3"):
            break
        print(c("❌ 输入无效，请输入 1、2 或 3", Color.YELLOW))

    if choice == "1":
        model_provider = "deepseek"
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 请设置 DEEPSEEK_API_KEY 环境变量")
            print("   方法1: export DEEPSEEK_API_KEY=your-key")
            print("   方法2: 创建 .env 文件写入 DEEPSEEK_API_KEY=your-key")
            return
    elif choice == "2":
        model_provider = "mimo"
        api_key = os.getenv("MIMO_API_KEY")
        if not api_key:
            print("❌ 请设置 MIMO_API_KEY 环境变量")
            print("   方法1: export MIMO_API_KEY=your-key")
            print("   方法2: 创建 .env 文件写入 MIMO_API_KEY=your-key")
            return
    else:
        model_provider = "openai"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ 请设置 OPENAI_API_KEY 环境变量")
            print("   方法1: export OPENAI_API_KEY=your-key")
            print("   方法2: 创建 .env 文件写入 OPENAI_API_KEY=your-key")
            return

    # 创建助手
    assistant = RAGAssistant(docs_dir="./docs", model_provider=model_provider, api_key=api_key)

    try:
        # 初始化
        assistant.initialize()

        # 交互循环
        print("💬 开始提问（输入 'quit' 退出，输入 'history' 查看历史）\n")

        history = []  # 保存 (问题, 回答) 历史

        while True:
            try:
                question = input(c("你：", Color.CYAN)).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break

            if not question:
                continue

            if question.lower() in ["quit", "exit", "退出"]:
                print("👋 再见！")
                break

            # 查看历史记录
            if question.lower() == "history":
                if not history:
                    print(c("（暂无历史记录）", Color.YELLOW))
                else:
                    for i, (q, a) in enumerate(history, 1):
                        print(f"{c('[' + str(i) + ']', Color.BOLD)} {c(q, Color.CYAN)}")
                        print(f"    {a}\n")
                continue

            # 提问
            answer = assistant.ask(question)
            print(f"\n{c('🤖 回答：', Color.GREEN)}{answer}\n")
            history.append((question, answer))
            print("-" * 50)

    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
