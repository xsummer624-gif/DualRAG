from langgraph.graph import StateGraph, END
from app.query_process.agent.state import QueryGraphState
# 导入所有节点函数
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_query_kg import node_query_kg
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp

# 初始化状态图
builder = StateGraph(QueryGraphState)

# 注册所有节点
builder.add_node("node_item_name_confirm", node_item_name_confirm)  # 确认商品
builder.add_node("node_search_embedding", node_search_embedding)  # 向量搜索
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
builder.add_node("node_web_search_mcp", node_web_search_mcp)
builder.add_node("node_rrf", node_rrf)  # 排序
builder.add_node("node_rerank", node_rerank)  # 重排
builder.add_node("node_answer_output", node_answer_output)  # 生成

builder.set_entry_point("node_item_name_confirm")

def route_after_node_item_name_confirm(state: QueryGraphState):
    if state.get("answer"):
        return "node_answer_output"
    return "node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"

builder.add_conditional_edges("node_item_name_confirm",
                              route_after_node_item_name_confirm)

builder.add_edge("node_search_embedding","node_rrf")
builder.add_edge("node_search_embedding_hyde","node_rrf")
builder.add_edge("node_web_search_mcp","node_rrf")
builder.add_edge("node_rrf","node_rerank")
builder.add_edge("node_rerank","node_answer_output")
builder.add_edge("node_answer_output",END)

query_app = builder.compile()