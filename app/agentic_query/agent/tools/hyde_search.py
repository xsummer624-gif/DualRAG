from langchain_core.tools import tool
from app.core.logger import logger
from app.query_process.agent.nodes.node_search_embedding_hyde import (
    step_1_create_hyde_doc,
    step_2_search_embedding_hyde,
)

MAX_CHARS = 300
MAX_CALLS = 2


def _truncate(content: str) -> str:
    return content[:MAX_CHARS] + "..." if len(content) > MAX_CHARS else content


def make_search_knowledge_base_enhanced():
    counter = {"n": 0}

    @tool
    def search_knowledge_base_enhanced(query: str, item_names: list[str] = None) -> list[dict]:
        """用 HyDE 增强方法在知识库中搜索。先生成一个假设的理想答案再用它检索，适合短查询或表述模糊的问题。query: 搜索问题; item_names: 限定商品名列表（可选）"""
        counter["n"] += 1
        if counter["n"] > MAX_CALLS:
            logger.warning(f"[Agent Tool] search_knowledge_base_enhanced 已达调用上限({MAX_CALLS}次),拒绝执行")
            return [{"info": f"search_knowledge_base_enhanced 已达调用上限({MAX_CALLS}次),请改用其他工具或直接基于已有信息回答"}]

        if isinstance(item_names, str):
            item_names = [item_names]

        logger.info(f"[Agent Tool] hyde_search #{counter['n']}/{MAX_CALLS}: query={query}, item_names={item_names}")

        hyde_doc = step_1_create_hyde_doc(query)
        logger.info(f"[Agent Tool] HyDE doc generated, length={len(hyde_doc)}")

        res = step_2_search_embedding_hyde(
            rewritten_query=query,
            hyde_doc=hyde_doc,
            item_names=item_names,
            top_k=5,
        )
        results = res[0] if res and len(res) > 0 else []
        for r in results:
            entity = r.get("entity", {})
            if "content" in entity:
                entity["_full_content"] = entity.get("content", "")
                entity["content"] = _truncate(entity.get("content", ""))
        logger.info(f"[Agent Tool] HyDE search found {len(results)} chunks")
        return results

    return search_knowledge_base_enhanced
