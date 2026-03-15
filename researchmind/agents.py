from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterator
from threading import Event
from time import perf_counter
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from researchmind import config
from researchmind.rag import RAGService


class SubGraphState(TypedDict):
    question: str
    question_index: int
    context: str
    source_refs: list[str]
    retrieval_count: int
    iteration_count: int
    literature_review: str
    data_analysis: str
    sub_answer: str
    node_timings: dict[str, float]


class MainGraphState(TypedDict):
    question: str
    sub_questions: list[str]
    sub_answers: list[dict]
    source_refs: list[str]
    final_prompt: str


StatusCallback = Callable[[str], None]


def _extract_text(response) -> str:
    text = getattr(response, "content", "")
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        collected: list[str] = []
        for item in text:
            if isinstance(item, str):
                collected.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    collected.append(str(item.get("text", "")))
                else:
                    candidate = item.get("text") or item.get("content") or item.get("output_text")
                    if candidate:
                        collected.append(str(candidate))
        return "".join(collected)

    alt_text = getattr(response, "text", None)
    if isinstance(alt_text, str) and alt_text:
        return alt_text

    additional = getattr(response, "additional_kwargs", None)
    if isinstance(additional, dict):
        candidate = (
            additional.get("text")
            or additional.get("content")
            or additional.get("output_text")
            or additional.get("reasoning_content")
        )
        if isinstance(candidate, str) and candidate:
            return candidate

    return str(text) if text else ""


class ResearchOrchestrator:
    def __init__(
        self,
        llm,
        rag_service: RAGService,
        rag_mode: str = config.DEFAULT_RAG_MODE,
        execution_mode: str = config.DEFAULT_EXECUTION_MODE,
    ):
        self._llm = llm
        self._rag_service = rag_service
        self._rag_mode = rag_mode
        self._execution_mode = execution_mode
        self._last_source_refs: list[str] = []

        if self._execution_mode == config.EXECUTION_MODE_EXTREME:
            self._max_subquestions = config.EXTREME_MAX_SUBQUESTIONS
            self._max_tool_calls = config.EXTREME_MAX_TOOL_CALLS
            self._max_iterations = config.EXTREME_MAX_ITERATIONS
            self._context_token_threshold = config.EXTREME_CONTEXT_TOKEN_THRESHOLD
            self._compressed_context_limit = config.EXTREME_COMPRESSED_CONTEXT_LIMIT
        elif self._execution_mode == config.EXECUTION_MODE_FAST:
            self._max_subquestions = config.FAST_MAX_SUBQUESTIONS
            self._max_tool_calls = config.FAST_MAX_TOOL_CALLS
            self._max_iterations = config.FAST_MAX_ITERATIONS
            self._context_token_threshold = config.FAST_CONTEXT_TOKEN_THRESHOLD
            self._compressed_context_limit = config.FAST_COMPRESSED_CONTEXT_LIMIT
        else:
            self._max_subquestions = config.MAX_SUBQUESTIONS
            self._max_tool_calls = config.MAX_TOOL_CALLS
            self._max_iterations = config.MAX_ITERATIONS
            self._context_token_threshold = config.CONTEXT_TOKEN_THRESHOLD
            self._compressed_context_limit = config.COMPRESSED_CONTEXT_LIMIT

        self._subgraph = self._build_subgraph()
        self._maingraph = self._build_main_graph()

    def get_last_source_refs(self) -> list[str]:
        return self._last_source_refs.copy()

    @staticmethod
    def _emit_status(callback: StatusCallback | None, message: str) -> None:
        if callback:
            callback(message)

    def _should_use_rag(self, question: str) -> bool:
        prompt = f"""
你是路由 Agent。请判断用户问题是否需要依赖“已上传文档内容”来回答。

规则：
- 如果问题是闲聊、常识、开放式观点、与已上传文档无关，返回 NO_RAG。
- 如果问题明确要求基于文档事实、章节、数据、引用、摘要，返回 USE_RAG。
- 仅输出一个标签：USE_RAG 或 NO_RAG。

用户问题：
{question}
"""
        try:
            response = self._llm.invoke(prompt)
            decision = _extract_text(response).strip().upper()
            return "USE_RAG" in decision and "NO_RAG" not in decision
        except Exception:
            return False

    def _split_subquestions(self, question: str) -> list[str]:
        prompt = f"""
你是问题拆解 Agent。请把用户研究问题拆解成 1 到 {self._max_subquestions} 个可并行检索的子问题。

要求：
- 每行一个子问题
- 子问题必须具体、可检索
- 仅输出子问题，不要额外解释

用户问题：
{question}
"""
        try:
            response = self._llm.invoke(prompt)
            text = _extract_text(response)
            lines = [line.strip(" -•\t0123456789.、") for line in text.splitlines()]
            parsed = [line.strip() for line in lines if line.strip()]
            if not parsed:
                return [question]
            return parsed[: self._max_subquestions]
        except Exception:
            return [question]

    def _compress_context(self, question: str, context: str) -> str:
        if len(context) <= self._context_token_threshold:
            return context

        prompt = f"""
请在不改变事实的前提下，压缩以下检索上下文，保留与问题最相关的信息。

用户问题：
{question}

上下文：
{context}

要求：
- 保留来源编号与关键数据
- 输出不超过 {self._compressed_context_limit} 字符
"""
        response = self._llm.invoke(prompt)
        compressed = _extract_text(response).strip()
        if not compressed:
            return context[: self._compressed_context_limit]
        return compressed[: self._compressed_context_limit]

    def _retrieve_with_budget(self, question: str) -> tuple[str, list[str], int, int]:
        iteration = 0
        tool_calls = 0
        retrieved_docs = []
        current_query = question

        while iteration < self._max_iterations and tool_calls < self._max_tool_calls:
            iteration += 1
            tool_calls += 1
            retrieved_docs = self._rag_service.retrieve(current_query)
            if retrieved_docs:
                break

            rewrite_prompt = f"""
请把下面的问题改写成更易检索的版本，只输出改写结果：
{current_query}
"""
            rewritten = _extract_text(self._llm.invoke(rewrite_prompt)).strip()
            current_query = rewritten or current_query

        context = self._rag_service.format_context(retrieved_docs)
        source_refs: list[str] = []
        for idx, document in enumerate(retrieved_docs, start=1):
            source = document.metadata.get("source", "unknown.source")
            page = document.metadata.get("page", "?")
            source_refs.append(f"[{idx}] {source} (page: {page})")

        return context, source_refs, tool_calls, iteration

    def _build_subgraph(self):
        def retrieve_context_node(state: SubGraphState) -> SubGraphState:
            started = perf_counter()
            context, refs, tool_calls, iterations = self._retrieve_with_budget(state["question"])
            node_timings = dict(state.get("node_timings", {}))
            node_timings["retrieve_context"] = round(perf_counter() - started, 2)
            return {
                "context": context,
                "source_refs": refs,
                "retrieval_count": tool_calls,
                "iteration_count": iterations,
                "node_timings": node_timings,
            }

        def compress_context_node(state: SubGraphState) -> SubGraphState:
            started = perf_counter()
            compressed = self._compress_context(state["question"], state.get("context", ""))
            node_timings = dict(state.get("node_timings", {}))
            node_timings["compress_context"] = round(perf_counter() - started, 2)
            return {
                "context": compressed,
                "node_timings": node_timings,
            }

        def literature_review_node(state: SubGraphState) -> SubGraphState:
            started = perf_counter()
            prompt = f"""
你是文献综述 Agent。你的任务是基于检索上下文，提炼该问题的研究现状与关键观点。

用户问题：
{state['question']}

检索上下文：
{state['context']}

请输出：
1) 研究主题
2) 关键发现（3-5条）
3) 争议点或空白点
"""
            review_text = _extract_text(self._llm.invoke(prompt))
            node_timings = dict(state.get("node_timings", {}))
            node_timings["literature_review"] = round(perf_counter() - started, 2)
            return {
                "literature_review": review_text,
                "node_timings": node_timings,
            }

        def data_analysis_node(state: SubGraphState) -> SubGraphState:
            started = perf_counter()
            prompt = f"""
你是数据分析 Agent。请根据文献综述与检索上下文，给出可执行的分析框架。

用户问题：
{state['question']}

文献综述结果：
{state['literature_review']}

检索上下文：
{state['context']}

请输出：
1) 可量化指标
2) 分析方法建议
3) 风险与偏差来源
"""
            analysis_text = _extract_text(self._llm.invoke(prompt))
            node_timings = dict(state.get("node_timings", {}))
            node_timings["data_analysis"] = round(perf_counter() - started, 2)
            return {
                "data_analysis": analysis_text,
                "node_timings": node_timings,
            }

        def paper_writing_node(state: SubGraphState) -> SubGraphState:
            started = perf_counter()
            prompt = f"""
你是论文写作 Agent。请整合前两个 Agent 的结果，为研究者提供结构化子结论。

用户问题：
{state['question']}

文献综述：
{state['literature_review']}

数据分析：
{state['data_analysis']}

检索上下文：
{state['context']}

要求：
- 使用中文
- 给出清晰小标题
- 末尾增加 2-3 条可执行建议
"""
            writing_text = _extract_text(self._llm.invoke(prompt))
            node_timings = dict(state.get("node_timings", {}))
            node_timings["paper_writing"] = round(perf_counter() - started, 2)
            return {
                "sub_answer": writing_text,
                "node_timings": node_timings,
            }

        workflow = StateGraph(SubGraphState)
        workflow.add_node("retrieve_context", retrieve_context_node)
        workflow.add_node("compress_context", compress_context_node)
        workflow.add_node("run_literature_review", literature_review_node)
        workflow.add_node("run_data_analysis", data_analysis_node)
        workflow.add_node("run_paper_writing", paper_writing_node)

        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", "compress_context")
        workflow.add_edge("compress_context", "run_literature_review")
        workflow.add_edge("run_literature_review", "run_data_analysis")
        workflow.add_edge("run_data_analysis", "run_paper_writing")
        workflow.add_edge("run_paper_writing", END)

        return workflow.compile()

    def _run_single_subgraph(
        self,
        index: int,
        question: str,
        status_callback: StatusCallback | None,
        stop_event: Event | None,
    ) -> dict:
        if stop_event and stop_event.is_set():
            return {
                "index": index,
                "question": question,
                "answer": "",
                "source_refs": [],
                "retrieval_count": 0,
                "iteration_count": 0,
            }

        start_time = perf_counter()

        initial_state: SubGraphState = {
            "question": question,
            "question_index": index,
            "context": "",
            "source_refs": [],
            "retrieval_count": 0,
            "iteration_count": 0,
            "literature_review": "",
            "data_analysis": "",
            "sub_answer": "",
            "node_timings": {},
        }

        result = self._subgraph.invoke(initial_state)
        elapsed = perf_counter() - start_time

        return {
            "index": index,
            "question": question,
            "answer": result.get("sub_answer", ""),
            "source_refs": result.get("source_refs", []),
            "retrieval_count": int(result.get("retrieval_count", 0)),
            "iteration_count": int(result.get("iteration_count", 0)),
            "elapsed_seconds": round(elapsed, 2),
            "node_timings": dict(result.get("node_timings", {})),
        }

    @staticmethod
    def _format_node_timing(node_timings: dict[str, float]) -> str:
        if not node_timings:
            return ""

        ordered_keys = [
            "retrieve_context",
            "compress_context",
            "literature_review",
            "data_analysis",
            "paper_writing",
        ]
        name_map = {
            "retrieve_context": "检索",
            "compress_context": "压缩",
            "literature_review": "综述",
            "data_analysis": "分析",
            "paper_writing": "写作",
        }
        items: list[str] = []
        for key in ordered_keys:
            if key in node_timings:
                items.append(f"{name_map[key]} {node_timings[key]:.2f}s")
        return "；".join(items)

    def _aggregate_subanswers(self, question: str, sub_answers: list[dict]) -> dict:
        if not sub_answers:
            fallback_prompt = f"""
请直接回答用户问题，使用中文，结构清晰并包含“可直接执行的下一步”。

用户问题：
{question}
"""
            return {"final_prompt": fallback_prompt, "source_refs": []}

        merged_subsections: list[str] = []
        merged_sources: list[str] = []
        seen = set()

        for item in sub_answers:
            merged_subsections.append(f"子问题 {item['index'] + 1}: {item['question']}\n{item['answer']}")
            for source in item.get("source_refs", []):
                if source not in seen:
                    seen.add(source)
                    merged_sources.append(source)

        final_prompt = f"""
你是论文写作 Agent。请综合多个子问题结果，生成最终研究答复。

原始问题：
{question}

子任务结果：
{'\n\n'.join(merged_subsections)}

要求：
- 使用中文
- 给出清晰小标题
- 输出“研究结论”“方法建议”“可直接执行的下一步”三个部分
- 若有引用来源，请在末尾给出“参考片段来源”
"""
        return {"final_prompt": final_prompt, "source_refs": merged_sources}

    def _build_main_graph(self):
        def rewrite_query_node(state: MainGraphState) -> MainGraphState:
            sub_questions = self._split_subquestions(state["question"])
            return {"sub_questions": sub_questions}

        def dispatch_subgraphs_node(state: MainGraphState) -> MainGraphState:
            sub_questions = state.get("sub_questions", [])
            if not sub_questions:
                return {"sub_answers": []}

            results: list[dict] = []
            max_workers = max(1, min(len(sub_questions), self._max_subquestions))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._run_single_subgraph, idx, sub_q, None, None)
                    for idx, sub_q in enumerate(sub_questions)
                ]
                for future in as_completed(futures):
                    results.append(future.result())

            results.sort(key=lambda item: item["index"])
            return {"sub_answers": results}

        def aggregate_node(state: MainGraphState) -> MainGraphState:
            return self._aggregate_subanswers(state["question"], state.get("sub_answers", []))

        workflow = StateGraph(MainGraphState)
        workflow.add_node("rewrite_query", rewrite_query_node)
        workflow.add_node("dispatch_subgraphs", dispatch_subgraphs_node)
        workflow.add_node("aggregate", aggregate_node)

        workflow.add_edge(START, "rewrite_query")
        workflow.add_edge("rewrite_query", "dispatch_subgraphs")
        workflow.add_edge("dispatch_subgraphs", "aggregate")
        workflow.add_edge("aggregate", END)

        return workflow.compile()

    def _answer_without_rag_prompt(self, question: str) -> str:
        return f"""
你是 ResearchMind 助手。当前问题无需依赖文档检索，请直接回答用户问题。

要求：
- 使用中文
- 简洁、自然
- 不要虚构文档引用

用户问题：
{question}
"""

    @staticmethod
    def _extract_chunk_text(chunk) -> str:
        return _extract_text(chunk)

    def _stream_prompt(self, prompt: str, stop_event: Event | None = None) -> Iterator[str]:
        emitted = False
        for chunk in self._llm.stream(prompt):
            if stop_event and stop_event.is_set():
                return
            text = self._extract_chunk_text(chunk)
            if text:
                emitted = True
                yield text

        if emitted:
            return

        fallback_text = ""
        try:
            fallback_text = _extract_text(self._llm.invoke(prompt)).strip()
        except Exception:
            fallback_text = ""

        if fallback_text:
            yield fallback_text
        else:
            yield "抱歉，模型暂时没有返回有效内容，请稍后重试。"

    def _run_main_graph_with_status(
        self,
        question: str,
        stop_event: Event | None,
        status_callback: StatusCallback | None,
    ) -> dict:
        split_start = perf_counter()
        self._emit_status(status_callback, "路由阶段：开始拆分子问题")
        sub_questions = self._split_subquestions(question)
        split_elapsed = perf_counter() - split_start
        self._emit_status(
            status_callback,
            f"路由阶段：生成 {len(sub_questions)} 个子问题 | 耗时 {split_elapsed:.2f}s",
        )
        if sub_questions:
            self._emit_status(status_callback, "检索阶段：开始并行执行子任务")

        results: list[dict] = []
        max_workers = max(1, min(len(sub_questions), self._max_subquestions))
        completed_count = 0
        retrieval_start = perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._run_single_subgraph, idx, sub_q, status_callback, stop_event)
                for idx, sub_q in enumerate(sub_questions)
            ]
            for future in as_completed(futures):
                if stop_event and stop_event.is_set():
                    break
                result = future.result()
                results.append(result)
                completed_count += 1
                node_timing_text = self._format_node_timing(result.get("node_timings", {}))
                node_timing_suffix = f"，节点：{node_timing_text}" if node_timing_text else ""
                self._emit_status(
                    status_callback,
                    (
                        f"检索阶段：完成 {completed_count}/{len(sub_questions)} | "
                        f"子任务 {result['index'] + 1}（检索 {result.get('retrieval_count', 0)} 次，"
                        f"迭代 {result.get('iteration_count', 0)} 轮，"
                        f"耗时 {result.get('elapsed_seconds', 0.0):.2f}s"
                        f"{node_timing_suffix}）"
                    ),
                )

        results.sort(key=lambda item: item["index"])
        retrieval_elapsed = perf_counter() - retrieval_start
        self._emit_status(status_callback, f"检索阶段：全部子任务完成 | 耗时 {retrieval_elapsed:.2f}s")
        self._emit_status(status_callback, "聚合阶段：正在汇总子任务结果")

        aggregate_start = perf_counter()
        prompt_state = self._aggregate_subanswers(question, results)
        aggregate_elapsed = perf_counter() - aggregate_start
        self._emit_status(status_callback, f"聚合阶段：汇总完成 | 耗时 {aggregate_elapsed:.2f}s")
        prompt_state["sub_answers"] = results
        return prompt_state

    def answer_stream(
        self,
        question: str,
        stop_event: Event | None = None,
        status_callback: StatusCallback | None = None,
    ) -> Iterator[str]:
        self._last_source_refs = []
        has_docs = self._rag_service.has_indexed_docs()

        if stop_event and stop_event.is_set():
            return

        if self._rag_mode == config.RAG_MODE_OFF:
            self._emit_status(status_callback, "RAG 路由：当前为禁用模式，直接回答")
            yield from self._stream_prompt(self._answer_without_rag_prompt(question), stop_event=stop_event)
            return

        if self._rag_mode == config.RAG_MODE_FORCE:
            if not has_docs:
                message = "当前是“强制 RAG”模式，但尚无已导入内容。请先上传文档或添加网页到 RAG 知识库。"
                yield message
                return
            use_rag = True
        else:
            route_start = perf_counter()
            self._emit_status(status_callback, "RAG 路由：判断是否需要检索")
            use_rag = self._should_use_rag(question) and has_docs
            route_elapsed = perf_counter() - route_start
            self._emit_status(status_callback, f"RAG 路由：判断完成 | 耗时 {route_elapsed:.2f}s")

        if not use_rag:
            self._emit_status(status_callback, "RAG 路由：无需检索，直接回答")
            yield from self._stream_prompt(self._answer_without_rag_prompt(question), stop_event=stop_event)
            return

        if stop_event and stop_event.is_set():
            return

        self._emit_status(status_callback, "RAG 路由：进入主图+子图并行执行")
        graph_start = perf_counter()
        graph_state = self._run_main_graph_with_status(question, stop_event, status_callback)
        graph_elapsed = perf_counter() - graph_start
        self._emit_status(status_callback, f"主流程阶段：主图+子图执行完成 | 耗时 {graph_elapsed:.2f}s")
        self._last_source_refs = graph_state.get("source_refs", [])

        final_prompt = graph_state.get("final_prompt", "")
        if not final_prompt.strip():
            final_prompt = self._answer_without_rag_prompt(question)

        self._emit_status(status_callback, "聚合阶段：开始生成最终回答")
        generation_start = perf_counter()
        yield from self._stream_prompt(final_prompt, stop_event=stop_event)
        generation_elapsed = perf_counter() - generation_start
        self._emit_status(status_callback, f"聚合阶段：最终回答生成完成 | 耗时 {generation_elapsed:.2f}s")

    def answer(self, question: str) -> tuple[str, list[str]]:
        answer = "".join(self.answer_stream(question, stop_event=None, status_callback=None))
        return answer, self.get_last_source_refs()
