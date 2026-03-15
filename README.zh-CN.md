# ResearchMind

ResearchMind 是一个面向科研与分析场景的 **Agentic RAG 研究助手** 项目。

它主要用于：
- 文献综述，
- 数据分析方案设计，
- 论文写作辅助。

当前版本使用阿里 DashScope 兼容接口模型（Qwen 系列）。

## 项目概览

ResearchMind 将多 Agent 协作与检索增强生成（RAG）结合，用于回答研究问题并尽量给出有依据的结果。

- 将复杂问题拆解为多个可执行子问题。
- 并行执行子任务。
- 从本地文档和网页中检索证据。
- 聚合子任务结果，生成结构化最终回答。

## 技术与框架（简要）

- **Python 3.12**：核心开发语言。
- **Streamlit**：项目 Web 界面（上传知识源 + 聊天问答）。
- **LangChain**：LLM 调用与提示词流程封装。
- **LangGraph**：多 Agent 状态图编排。
- **ChromaDB**：本地向量数据库持久化。
- **HuggingFace Embeddings**（`all-MiniLM-L6-v2`）：向量检索嵌入。
- **PyPDF / python-docx / BeautifulSoup**：PDF、DOCX、网页内容解析与入库。

## 架构亮点

- **主图（LangGraph）**：
  - 路由决策，
  - 问题拆解，
  - 子图并行派发，
  - 最终聚合。
- **子图流程**：
  - 检索，
  - 上下文压缩，
  - 文献综述，
  - 数据分析，
  - 写作整合。
- **RAG 流水线**：
  - Parent/Child 双层切分，
  - dense + sparse 混合召回，
  - 轻量 rerank，
  - 引用来源追踪。

## 项目结构

```text
ResearchMind/
├── app.py
├── main.py
├── pyproject.toml
├── README.md
├── README.zh-CN.md
└── researchmind/
    ├── config.py
    ├── llm.py
    ├── rag.py
    ├── agents.py
    └── ui.py
```

## 使用 `uv` 快速运行

### 1）创建虚拟环境并安装依赖

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 2）配置 API Key

```bash
export DASHSCOPE_API_KEY="你的key"
```

也兼容：

```bash
export OPENAI_API_KEY="你的key"
```

### 3）启动项目

```bash
uv run streamlit run app.py
```

终端会显示本地 Streamlit 访问地址，打开即可使用。

## 典型使用流程

1. 上传一个或多个 `.pdf` / `.docx` 文件，或添加网页 URL。
2. 导入到 RAG 知识库。
3. 输入研究问题发起问答。
4. 查看最终回答及参考片段来源。

## 当前限制

- 暂不支持 `.doc`（仅支持 `.docx`）。
- 当前仅支持 DashScope 兼容接口。
- 向量数据默认持久化在 `data/chroma/`。

## 项目特性

- 覆盖从数据入库、检索增强、Agent 编排到交互界面的端到端 LLM 应用流程。
- 基于图编排的多 Agent 任务拆解、并行执行与结果聚合机制。
- 面向实际 RAG 场景的混合检索与来源可追溯能力。
