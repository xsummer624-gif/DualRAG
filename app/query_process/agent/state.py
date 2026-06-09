from typing_extensions import TypedDict
from typing import List
import copy


class QueryGraphState(TypedDict):
    """查询流程全局状态"""
    session_id: str           # 会话唯一标识
    original_query: str       # 用户原始问题

    # 检索中间数据
    embedding_chunks: list    # 向量检索切片
    hyde_embedding_chunks: list  # HyDE 检索切片
    kg_chunks: list           # 图谱检索切片
    web_search_docs: list     # 网络搜索文档

    # 排序中间数据
    rrf_chunks: list          # RRF 融合后切片
    reranked_docs: list       # 重排序后 Top-K 文档

    # 生成中间数据
    prompt: str               # 组装好的 Prompt
    answer: str               # 最终答案

    # 辅助信息
    item_names: List[str]     # 提取的商品名
    rewritten_query: str      # 改写后的问题
    history: list             # 历史对话
    is_stream: bool           # 是否流式输出


# 默认初始状态
graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "kg_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
}


def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态，支持关键字覆盖"""
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    """获取一份全新的默认状态副本"""
    return copy.deepcopy(graph_default_state)
