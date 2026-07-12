import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from app.core.logger import logger
from app.conf.lm_config import lm_config
from app.utils.task_utils import (
    update_task_status, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED,
    set_task_result,
)
from app.utils.sse_utils import push_to_session, SSEEvent
from app.clients.mongo_history_utils import save_chat_message, get_recent_messages
from app.evaluation.ragas_metrics import full_evaluation
from app.query_process.agent.nodes.node_item_name_confirm import (
    step_3_extract_info,
    step_4_vectorize_and_query,
    step_5_align_item_names,
)
from app.agentic_query.agent.tools.kb_search import make_search_knowledge_base
from app.agentic_query.agent.tools.hyde_search import make_search_knowledge_base_enhanced
from app.agentic_query.agent.tools.web_search import make_search_web

BASE_PROMPT = """你是掌柜智库的智能客服助手，专门为用户解答产品使用、技术规格、故障排查等问题。
你的知识主要来自本地知识库（产品手册、说明书），必要时可联网补充实时信息。

## 可用工具

### search_knowledge_base(query, item_names)
向量检索本地知识库。参数：
- query: 传完整的问题语句（不要只传关键词），例如"HAK180烫金机怎么调节温度"
- item_names: 商品名列表，用于缩小搜索范围。下方"主题识别结果"给出了已确认的商品名，
  你**必须使用**这些商品名作为 item_names 参数。如果主题识别结果为空（未识别到商品名），
  则传 None 或不传，系统会做全库搜索。

### search_knowledge_base_enhanced(query, item_names)
HyDE 增强检索，会先生成假设答案再用它检索。参数同上。
适用场景：问题表述模糊、短查询、或普通检索返回空结果时，换用此工具。

### search_web(query)
联网搜索，获取知识库可能没有的实时信息（最新型号、新闻、价格等）。参数：
- query: 搜索关键词，简洁明确即可

## 搜索策略

### 工具选择
- 优先使用知识库工具（search_knowledge_base）。知识库覆盖了产品的完整手册，是主要信息来源。
- 以下情况使用 search_knowledge_base_enhanced：问题模糊、短查询、或普通检索返回空。
- 以下情况使用 search_web：问题涉及实时信息（新型号、价格、新闻）、或知识库工具返回空且问题不在产品手册范畴内。

### 搜索次数
- 每个工具最多可调用 2 次（系统强制限制，超过会返回"已达上限"提示，无法再执行）。
- 如果某工具返回"已达调用上限"提示，立即换用其他工具或直接基于已有信息回答，不要再尝试调用该工具。
- 如果第一次检索结果已经能完整回答问题，无需凑满次数，直接生成最终回答。
- 如果某个工具返回空，换一个工具重试（不要用相同参数重复调同一个工具）。
- 禁止用完全相同的参数重复调用任何工具。

### 结果充分性判断
搜索结果"足够"的标准：能直接回答用户问题的核心诉求。
- 如果结果包含了用户问的具体参数/步骤/方法 → 足够，立即回答。
- 如果结果只涉及部分，且缺失部分可能是另一个产品或另一个话题 → 足够，回答已知部分，说明未覆盖的部分。
- 如果结果完全空白或明显不相关 → 不够，换工具或换 query 重试（受每工具 2 次上限约束）。

## 回答规范

### 结构
1. 先给直接结论（用户最想知道的一句话答案）。
2. 再展开细节（参数、步骤、注意事项）。
3. 如有多个方面，用分点列出。
4. 最后附来源标注，格式：[来源: 知识库/联网]。

### 无结果时的回答
当所有工具都返回空时：
- 不要编造。明确告知"知识库中未找到相关信息"。
- 如果自身知识能回答，可以回答但必须标注"以下为通用知识，非知识库内容"。
- 建议用户细化问题或提供产品型号。

### 图片
如果检索到的文档片段中包含图片URL，且图片对理解答案有帮助（外观、接线图、示意图等），
在答案最后追加独立的图片区块，格式严格如下：
【图片】
<URL1>
<URL2>
（每行一个URL；无图片则不要输出此区块）

### 语言与风格
- 中文回答。
- 专业但易懂，避免不必要的技术黑话。
- 长度适中：事实型问题简洁回答，步骤型问题给完整步骤。"""


def build_system_prompt(db_item_names: list, rewritten_query: str) -> str:
    ctx = "\n\n## 主题识别结果（预处理，已执行，无需你再调用工具识别）"
    if db_item_names:
        ctx += f"\n已确认商品名: {', '.join(db_item_names)}"
        ctx += f"\n→ 调用知识库工具时，item_names 参数必须传上述商品名。"
    else:
        ctx += f"\n未识别到具体商品名。"
        ctx += f"\n→ 调用知识库工具时，item_names 传 None（全库搜索）。如果用户明确提到了产品型号，可以用你从问题中提取的名字作为 item_names。"
    if rewritten_query:
        ctx += f"\n改写后的问题: {rewritten_query}"
        ctx += f"\n→ 建议用改写后的问题作为 search_knowledge_base 的 query 参数，语义更完整。"
    return BASE_PROMPT + ctx


def _run_topic_identification(user_query: str, history: list) -> dict:
    """强制运行主题识别：step3 LLM提取 → step4 向量对齐 → step5 确认"""
    result = {"db_item_names": [], "rewritten_query": user_query, "error": None}
    try:
        extracted = step_3_extract_info(user_query, history)
        rewritten_query = extracted.get("rewritten_query", user_query)
        extracted_names = extracted.get("item_names", [])
        result["rewritten_query"] = rewritten_query

        if extracted_names:
            vector_results = step_4_vectorize_and_query(extracted_names)
            align_result = step_5_align_item_names(vector_results)
            confirmed = align_result.get("confirmed_item_names", []) or align_result.get("options", [])
            if confirmed:
                result["db_item_names"] = confirmed
                logger.info(f"[主题识别] 商品名对齐成功: {confirmed}")
            else:
                logger.info(f"[主题识别] 未找到匹配的数据库商品名，不传 item_names 进行全库搜索")
                result["db_item_names"] = []
        else:
            logger.info("[主题识别] 未提取到商品名")
    except Exception as e:
        logger.warning(f"[主题识别] 异常(不影响主流程): {e}")
        result["error"] = str(e)
    return result


def extract_agent_chain(messages) -> list:
    chain = []
    calls_by_id = {}
    for msg in messages:
        tc = getattr(msg, "tool_calls", None) or []
        for call in tc:
            call_id = call.get("id", "")
            entry = {"tool": call.get("name", ""), "input": call.get("args", {}), "output": None}
            calls_by_id[call_id] = entry
            chain.append(entry)
    for msg in messages:
        if hasattr(msg, "name") and getattr(msg, "type", "") == "tool":
            call_id = getattr(msg, "tool_call_id", "")
            content = msg.content or ""
            content = content[:500] if len(content) > 500 else content
            if call_id and call_id in calls_by_id:
                calls_by_id[call_id]["output"] = content
            else:
                for entry in reversed(chain):
                    if entry["output"] is None:
                        entry["output"] = content
                        break
    return chain


_model = None


def get_model():
    global _model
    if _model is None:
        _model = ChatOpenAI(
            model=lm_config.llm_model,
            openai_api_key=lm_config.api_key,
            openai_api_base=lm_config.base_url,
            temperature=lm_config.llm_temperature,
            extra_body={"enable_thinking": False},
        )
    return _model


def build_agent_executor():
    """每次调用构建一组全新的工具实例（带独立计数器）+ 新的 agent，
    实现 per-task 的工具调用次数隔离。"""
    tools = [
        make_search_knowledge_base(),
        make_search_knowledge_base_enhanced(),
        make_search_web(),
    ]
    return create_react_agent(
        model=get_model(),
        tools=tools,
        name="agentic_rag_agent",
    )


async def run_agentic_graph(state: dict) -> dict:
    session_id = state["session_id"]
    user_query = state["original_query"]
    is_stream = state.get("is_stream", False)
    logger.info(f"[AgenticRAG] 开始处理: session={session_id}, query={user_query}")

    try:
        all_history = get_recent_messages(session_id, limit=10)
        history = [m for m in all_history if m.get("role") in ("user", "assistant", "bot")]
    except Exception:
        history = []

    topic = _run_topic_identification(user_query, history)
    db_item_names = topic["db_item_names"]
    rewritten_query = topic["rewritten_query"]

    system_prompt = build_system_prompt(db_item_names, rewritten_query)
    messages = [SystemMessage(content=system_prompt)]
    for msg in history:
        role = msg.get("role", "")
        text = msg.get("text", "")
        if role == "user":
            messages.append(HumanMessage(content=text))
        elif role in ("assistant", "bot"):
            messages.append(AIMessage(content=text))
    messages.append(HumanMessage(content=user_query))

    try:
        agent = build_agent_executor()
        config = {"recursion_limit": 16, "configurable": {"recursion_limit": 16}}

        agent_messages = []
        answer = ""

        if is_stream:
            async for event in agent.astream_events(
                {"messages": messages},
                config,
                version="v2",
            ):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk:
                        delta = getattr(chunk, "content", "") or ""
                        if delta:
                            answer += delta
                            push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
                elif kind == "on_chain_end":
                    output = event["data"].get("output", {})
                    if isinstance(output, dict) and "messages" in output:
                        agent_messages = output["messages"]

            if not answer and agent_messages:
                answer = getattr(agent_messages[-1], "content", "") or ""
        else:
            result = await agent.ainvoke({"messages": messages}, config)
            agent_messages = result["messages"]
            answer = agent_messages[-1].content

        agent_chain = extract_agent_chain(agent_messages)

        topic_step = {
            "tool": "topic_identification",
            "input": {"query": user_query, "identified_names": db_item_names},
            "output": json.dumps({"db_item_names": db_item_names, "rewritten_query": rewritten_query}, ensure_ascii=False),
        }
        agent_chain = [topic_step] + agent_chain

        contexts = []
        for msg in agent_messages:
            if hasattr(msg, "name") and getattr(msg, "type", "") == "tool":
                tool_name = msg.name
                try:
                    data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if not isinstance(data, list):
                        continue
                    if tool_name in ("search_knowledge_base", "search_knowledge_base_enhanced"):
                        for d in data:
                            if isinstance(d, dict):
                                entity = d.get("entity", d)
                                full = entity.get("_full_content") or entity.get("content", "")
                                if full:
                                    contexts.append(full)
                    elif tool_name == "search_web":
                        for d in data:
                            if isinstance(d, dict):
                                content = d.get("content", "")
                                if content:
                                    contexts.append(content)
                except (json.JSONDecodeError, TypeError):
                    pass

        state["answer"] = answer
        state["agent_chain"] = agent_chain
        set_task_result(session_id, "answer", answer)
        set_task_result(session_id, "agent_chain", json.dumps(agent_chain, ensure_ascii=False))
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)

        try:
            save_chat_message(
                session_id=session_id, role="user", text=user_query,
                rewritten_query=rewritten_query, item_names=db_item_names,
            )
            save_chat_message(
                session_id=session_id, role="assistant", text=answer,
                item_names=db_item_names,
            )
        except Exception as e:
            logger.warning(f"[AgenticRAG] 保存历史失败: {e}")

        rag_scores_final = None
        try:
            scores = full_evaluation(question=user_query, answer=answer, contexts=contexts)
            scores["session_id"] = session_id
            rag_scores_final = scores
            set_task_result(session_id, "rag_scores", json.dumps(scores, ensure_ascii=False))
            try:
                save_chat_message(session_id=session_id, role="system", text=f"[RAG评估] {json.dumps(scores, ensure_ascii=False)}")
            except Exception:
                pass
            logger.info(f"[AgenticRAG] RAG评估: {scores}")
        except Exception as e:
            logger.warning(f"[AgenticRAG] 评估异常(不影响): {e}")

        if is_stream:
            try:
                push_to_session(session_id, SSEEvent.FINAL, {
                    "answer": answer, "status": "completed", "image_urls": [],
                    "agent_chain": agent_chain, "rag_scores": rag_scores_final,
                })
            except Exception:
                pass

        logger.info(f"[AgenticRAG] 处理完成: session={session_id}")
        return state

    except Exception as e:
        logger.error(f"[AgenticRAG] 流程异常: {e}", exc_info=True)
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
        state["answer"] = f"处理失败: {e}"
        return state
