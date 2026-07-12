import json
from langchain_core.tools import tool
from app.core.logger import logger
from app.query_process.agent.nodes.node_web_search_mcp import mcp_call_streamable

MAX_CALLS = 2


def make_search_web():
    counter = {"n": 0}

    @tool
    async def search_web(query: str) -> list[dict]:
        """搜索互联网获取最新的外部信息。适用于需要实时数据、新闻、最新规格参数或本地知识库没有覆盖的内容。query: 搜索关键词"""
        counter["n"] += 1
        if counter["n"] > MAX_CALLS:
            logger.warning(f"[Agent Tool] search_web 已达调用上限({MAX_CALLS}次),拒绝执行")
            return [{"info": f"search_web 已达调用上限({MAX_CALLS}次),请改用其他工具或直接基于已有信息回答"}]

        logger.info(f"[Agent Tool] search_web #{counter['n']}/{MAX_CALLS}: query={query}")

        result = await mcp_call_streamable(query)

        web_documents = []
        if result and result.content:
            try:
                raw_text = result.content[0].text
                pages = json.loads(raw_text).get("pages", [])
                for p in pages:
                    content = p.get("content", "")
                    if len(content) > 300:
                        content = content[:300] + "..."
                    web_documents.append({
                        "title": p.get("title", ""),
                        "url": p.get("url", ""),
                        "content": content,
                    })
            except Exception as e:
                logger.error(f"[Agent Tool] Web search parse failed: {e}")

        logger.info(f"[Agent Tool] Web search found {len(web_documents)} results")
        return web_documents

    return search_web
