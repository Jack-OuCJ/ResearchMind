from __future__ import annotations

import streamlit as st

from researchmind import config
from researchmind.agents import ResearchOrchestrator
from researchmind.llm import create_ali_chat_model
from researchmind.rag import RAGService


def _resolve_max_tokens(execution_mode: str) -> int:
    if execution_mode == config.EXECUTION_MODE_EXTREME:
        return config.EXTREME_MAX_OUTPUT_TOKENS
    if execution_mode == config.EXECUTION_MODE_FAST:
        return config.FAST_MAX_OUTPUT_TOKENS
    return config.MAX_OUTPUT_TOKENS


def _init_state() -> None:
    if "rag_service" not in st.session_state:
        st.session_state.rag_service = RAGService()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_name" not in st.session_state:
        st.session_state.model_name = config.DEFAULT_MODEL
    if "rag_mode" not in st.session_state:
        st.session_state.rag_mode = config.DEFAULT_RAG_MODE
    if "font_size" not in st.session_state:
        st.session_state.font_size = config.DEFAULT_FONT_SIZE
    if "execution_mode" not in st.session_state:
        st.session_state.execution_mode = config.DEFAULT_EXECUTION_MODE
    if "orchestrator" not in st.session_state:
        max_tokens = _resolve_max_tokens(st.session_state.execution_mode)
        llm = create_ali_chat_model(st.session_state.model_name, max_tokens=max_tokens)
        st.session_state.orchestrator = ResearchOrchestrator(
            llm,
            st.session_state.rag_service,
            rag_mode=st.session_state.rag_mode,
            execution_mode=st.session_state.execution_mode,
        )
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False


def _rebuild_orchestrator(model_name: str, rag_mode: str, execution_mode: str) -> None:
    max_tokens = _resolve_max_tokens(execution_mode)
    llm = create_ali_chat_model(model_name, max_tokens=max_tokens)
    st.session_state.orchestrator = ResearchOrchestrator(
        llm,
        st.session_state.rag_service,
        rag_mode=rag_mode,
        execution_mode=execution_mode,
    )
    st.session_state.model_name = model_name
    st.session_state.rag_mode = rag_mode
    st.session_state.execution_mode = execution_mode


def _on_model_or_rag_change() -> None:
    _rebuild_orchestrator(
        st.session_state.model_name,
        st.session_state.rag_mode,
        st.session_state.execution_mode,
    )


def _apply_font_size(font_size: str) -> None:
    size_value = config.FONT_SIZE_VALUES.get(font_size, config.FONT_SIZE_VALUES[config.DEFAULT_FONT_SIZE])
    st.markdown(
        f"""
<style>
[data-testid="stChatMessage"] p,
[data-testid="stMarkdownContainer"] p,
[data-testid="stChatInput"] textarea {{
  font-size: {size_value};
}}

/* 让停止按钮与底部输入框更紧凑 */
div[data-testid="stButton"] {{
    margin-bottom: 0 !important;
}}

div[data-testid="stChatInput"] {{
    margin-top: 0.15rem !important;
}}

.rm-idle-wrap {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    opacity: 0.88;
}}

.rm-idle-spinner {{
    width: 0.95rem;
    height: 0.95rem;
    border: 2px solid currentColor;
    border-top-color: transparent;
    border-radius: 50%;
    display: inline-block;
    animation: rm-idle-spin 0.8s linear infinite;
}}

@keyframes rm-idle-spin {{
    to {{
        transform: rotate(360deg);
    }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


def run_app() -> None:
    st.set_page_config(page_title="ResearchMind", page_icon="📓", layout="wide")
    _init_state()

    with st.sidebar:
        st.header("文件库")

        st.selectbox(
            "阿里模型",
            options=config.SUPPORTED_MODELS,
            key="model_name",
            on_change=_on_model_or_rag_change,
        )

        st.selectbox(
            "RAG 模式",
            options=config.RAG_MODES,
            key="rag_mode",
            on_change=_on_model_or_rag_change,
            format_func=lambda mode: config.RAG_MODE_LABELS.get(mode, mode),
        )

        st.selectbox(
            "执行速度",
            options=config.EXECUTION_MODES,
            key="execution_mode",
            on_change=_on_model_or_rag_change,
            format_func=lambda mode: config.EXECUTION_MODE_LABELS.get(mode, mode),
        )

        selected_font_size = st.selectbox(
            "字体大小",
            options=config.FONT_SIZE_OPTIONS,
            key="font_size",
            format_func=lambda item: config.FONT_SIZE_LABELS.get(item, item),
        )

        uploads = st.file_uploader(
            "上传文档（支持 PDF / Word .docx，可多选）",
            type=["pdf", "docx"],
            accept_multiple_files=True,
        )
        if st.button("导入到 RAG 知识库", use_container_width=True):
            if not uploads:
                st.warning("请先选择至少一个文档文件。")
            else:
                try:
                    with st.spinner("正在解析并写入向量库..."):
                        total_chunks = 0
                        for upload in uploads:
                            total_chunks += st.session_state.rag_service.ingest_file(upload.name, upload.getvalue())
                    st.success(f"导入完成，共写入 {total_chunks} 个文本分块。")
                except Exception as error:
                    st.error(f"文档导入失败：{error}")

        web_url = st.text_input("添加网页 URL", placeholder="https://example.com/article")
        if st.button("添加网页到知识库", use_container_width=True):
            if not web_url.strip():
                st.warning("请输入网页 URL。")
            else:
                try:
                    with st.spinner("正在抓取网页并写入向量库..."):
                        total_chunks = st.session_state.rag_service.ingest_web_url(web_url)
                    st.success(f"网页导入完成，共写入 {total_chunks} 个文本分块。")
                except Exception as error:
                    st.error(f"网页导入失败：{error}")

        st.subheader("已导入来源")
        sources = st.session_state.rag_service.list_knowledge_sources()
        if not sources:
            st.caption("暂无来源，请先上传文档或添加网页。")
        else:
            for source in sources:
                st.markdown(f"- {source}")

    _apply_font_size(st.session_state.font_size)

    st.title("📓 ResearchMind")
    st.caption("多 Agent 协作与 RAG 驱动的研究助手")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            source_refs = message.get("source_refs", [])
            if source_refs:
                st.markdown("\n".join(["\n**参考片段来源**"] + [f"- {item}" for item in source_refs]))

    user_query = st.chat_input(
        "输入你的研究问题..."
    )

    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.is_generating = True

    with st.chat_message("assistant"):
        working_placeholder = st.empty()
        rag_log_placeholder = st.empty()
        rag_logs: list[str] = []

        def _render_rag_logs() -> None:
            if not rag_logs:
                rag_log_placeholder.empty()
                return
            recent_logs = rag_logs[-12:]
            rag_log_placeholder.markdown(
                "\n".join(["**RAG 执行日志**"] + [f"- {line}" for line in recent_logs])
            )

        def _status_callback(message: str) -> None:
            clean_message = (message or "").strip()
            if not clean_message:
                return
            rag_logs.append(clean_message)
            _render_rag_logs()

        working_placeholder.markdown(
            """
<div class="rm-idle-wrap" aria-live="polite">
  <span class="rm-idle-spinner"></span>
  <span>思考中…</span>
</div>
""",
            unsafe_allow_html=True,
        )
        try:
            stream_started = False

            def _answer_stream_with_status_clear():
                nonlocal stream_started
                for chunk in st.session_state.orchestrator.answer_stream(user_query, status_callback=_status_callback):
                    visible_chunk = str(chunk or "").replace("\u200b", "").strip()
                    if (not stream_started) and visible_chunk:
                        working_placeholder.empty()
                        stream_started = True
                    yield chunk

            answer = st.write_stream(_answer_stream_with_status_clear())
            if not isinstance(answer, str):
                answer = "" if answer is None else str(answer)
            if not answer.strip():
                answer = "抱歉，这次没有拿到有效回复，请重试一次。"
                st.markdown(answer)
            source_refs = st.session_state.orchestrator.get_last_source_refs()
        except Exception as error:
            working_placeholder.empty()
            error_text = str(error)
            error_lower = error_text.lower()
            if "rustbindingsapi" in error_lower or "has no attribute 'bindings'" in error_lower:
                answer = (
                    "向量库依赖版本不兼容（RustBindingsAPI.bindings）。"
                    "请安装 chromadb<1.0（如 chromadb>=0.5.23,<1.0.0）后重启应用。"
                )
            else:
                answer = "请求超时或模型服务暂时不可用，请稍后重试，或切换到 qwen-turbo。"
            source_refs = []
            st.error(f"{answer}\n\n错误详情: {error}")
            st.session_state.messages.append({"role": "assistant", "content": answer, "source_refs": []})
            st.session_state.is_generating = False
            return
        working_placeholder.empty()
        _render_rag_logs()
        if source_refs:
            st.markdown("\n".join(["\n**参考片段来源**"] + [f"- {item}" for item in source_refs]))

    st.session_state.is_generating = False
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "source_refs": source_refs,
        }
    )
