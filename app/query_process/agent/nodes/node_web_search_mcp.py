import sys
import json
import asyncio

from dotenv import load_dotenv

from app.query_process.agent import state
from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from agents.mcp import MCPServerSse, MCPServerStreamableHttp
from app.core.logger import logger
from app.utils.task_utils import add_done_task, add_running_task

DASHSCOPE_BASE_URL_STREAMABLE = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key

async def mcp_call_streamable(query):
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": DASHSCOPE_BASE_URL_STREAMABLE,
            "headers": {"Authorization": DASHSCOPE_API_KEY},
            "timeout": 10,
        },
        max_retry_attempts=3
    )
    try:
        await search_mcp.connect()
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": query,
                "count":5
            }
        )
        return result
    finally:
        try:
            await search_mcp.cleanup()
        except Exception:
            pass


def node_web_search_mcp(state):
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    query = state.get("rewritten_query")
    if not query:
        logger.warning("MCP 搜索跳过：无 rewritten_query")
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
        return {"web_search_docs": []}

    logger.info(f"启动异步 MCP 调用，Query: {query}")
    result = asyncio.run(mcp_call_streamable(query))

    web_documents = []
    if result and result.content:
        try:
            raw_text = result.content[0].text
            pages = json.loads(raw_text).get("pages", [])
            for p in pages:
                web_documents.append({
                    "title": p.get("title", ""),
                    "url": p.get("url", ""),
                    "content": p.get("content", ""),
                })
        except Exception as e:
            logger.error(f"MCP 结果解析失败: {e}")

    logger.info(f"MCP 搜索结果数量: {len(web_documents)}")
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"web_search_docs": web_documents}


if __name__ == '__main__':
    load_dotenv()
    # 测试代码：单独运行该文件时，验证MCP搜索功能是否正常
    print("\n" + "=" * 50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("=" * 50)

    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": True
    }

    try:
        # 调用MCP搜索节点函数，执行测试
        result_state = node_web_search_mcp(test_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get('web_search_docs', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")