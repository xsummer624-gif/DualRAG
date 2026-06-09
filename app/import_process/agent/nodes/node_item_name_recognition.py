import os
import sys
from typing import List, Dict, Any, Tuple

# Milvus 客户端 + Schema 定义
from pymilvus import MilvusClient, DataType
# LangChain 消息类型（标准化大模型对话格式）
from langchain_core.messages import SystemMessage, HumanMessage

# 项目内部模块
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.utils.normalize_sparse_vector import normalize_sparse_vector
from app.utils.task_utils import add_running_task, add_done_task
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.core.logger import logger
from app.core.load_prompt import load_prompt

# ============================================================
# 配置参数
# ============================================================

# LLM 识别商品名时取前 K 个切片作为上下文
DEFAULT_ITEM_NAME_CHUNK_K = 5

# 单个切片内容截断长度（防止单个 Chunk 占满上下文）
SINGLE_CHUNK_CONTENT_MAX_LEN = 800

# LLM 上下文总字符数上限（适配主流大模型输入限制）
CONTEXT_TOTAL_MAX_CHARS = 2500


# ============================================================
# 步骤 1：提取并校验输入数据
# ============================================================

def step_1_get_inputs(state: ImportGraphState) -> Tuple[str, List[Dict]]:
    """
    从流程状态中提取文件标题和文本切片，做多层空值兜底
    上游：node_document_split（state 中有 file_title、chunks）
    :param state: 流程状态对象
    :return: (文件标题, 切片列表)
    """
    # 多层兜底：file_title → file_name → 从第一个切片提取
    file_title = state.get("file_title", "") or state.get("file_name", "")

    chunks = state.get("chunks") or []

    # 二次兜底：file_title 仍为空，从第一个有效切片中抠
    if not file_title:
        if chunks and isinstance(chunks[0], dict):
            file_title = chunks[0].get("file_title", "")
            logger.warning("state中无有效file_title，已从第一个切片中提取兜底标题")

    if not file_title:
        logger.warning("state中缺少file_title，后续大模型识别可能精度下降")

    # chunks 类型校验
    if not isinstance(chunks, list) or not chunks:
        logger.warning("state中chunks为空或非列表类型，无法进行商品名称识别")
        return file_title, []

    logger.info(f"步骤1：输入校验完成，获取到{len(chunks)}个有效文本切片")
    return file_title, chunks


# ============================================================
# 步骤 2：构建 LLM 识别上下文
# ============================================================

def step_2_build_context(
    chunks: List[Dict],
    k: int = DEFAULT_ITEM_NAME_CHUNK_K,
    max_chars: int = CONTEXT_TOTAL_MAX_CHARS
) -> str:
    """
    截取前 K 个切片的结构化内容，拼接成 LLM 可阅读的上下文
    目的：文档开头几个章节通常是产品概述，包含商品名信息密度最高
    :param chunks: 切片列表
    :param k: 最多取几个切片
    :param max_chars: 上下文总字符数上限
    :return: 格式化后的上下文字符串
    """
    if not chunks:
        return ""

    parts: List[str] = []
    total_chars = 0

    for idx, chunk in enumerate(chunks[:k]):
        if not isinstance(chunk, dict):
            continue

        chunk_title = chunk.get("title", "").strip()
        chunk_content = chunk.get("content", "").strip()

        # 跳过完全空白切片
        if not (chunk_title or chunk_content):
            continue

        # 单切片截断
        if len(chunk_content) > SINGLE_CHUNK_CONTENT_MAX_LEN:
            chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN]

        # 结构化格式化：带序号 + 标题 + 内容
        piece = f"【切片{idx + 1}】\n标题：{chunk_title}\n内容：{chunk_content}"
        parts.append(piece)
        total_chars += len(piece)

        if total_chars > max_chars:
            logger.info(f"上下文总字符数即将超限（{max_chars}），已停止拼接后续切片")
            break

    context = "\n\n".join(parts).strip()
    final_context = context[:max_chars]
    logger.info(f"步骤2：上下文构建完成，最终长度{len(final_context)}字符")
    return final_context


# ============================================================
# 步骤 3：调用 LLM 识别商品名称
# ============================================================

def step_3_call_llm(file_title: str, context: str) -> str:
    """
    调用大模型从上下文和文件名中识别最核心的商品名称
    三层兜底：LLM 成功 → file_title → 空字符串
    :param file_title: 文件标题（兜底值）
    :param context: 步骤2构建的上下文
    :return: 识别的商品名称字符串
    """
    logger.info("开始执行步骤3：调用大模型识别商品名称")

    if not context:
        logger.warning("上下文为空，跳过大模型调用，直接使用文件标题作为商品名称")
        return file_title

    try:
        # 加载提示词模板
        human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
        system_prompt = load_prompt("product_recognition_system")

        # 获取 LLM 客户端（纯文本模式，不需 JSON 结构）
        llm = get_llm_client(json_mode=False)
        if not llm:
            logger.error("大模型客户端获取失败，使用文件标题兜底")
            return file_title

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        resp = llm.invoke(messages)
        item_name = getattr(resp, "content", "").strip()

        # 清洗返回值：去空格、换行、制表符
        item_name = item_name.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

        if not item_name:
            logger.warning("大模型返回空内容，使用文件标题作为商品名称兜底")
            return file_title

        logger.info(f"步骤3：大模型识别商品名称成功，结果为：{item_name}")
        return item_name

    except Exception as e:
        logger.error(f"步骤3：大模型调用失败，原因：{str(e)}", exc_info=True)
        return file_title


# ============================================================
# 步骤 4：回填商品名称到 state 和所有切片
# ============================================================

def step_4_update_chunks(state: ImportGraphState, chunks: List[Dict], item_name: str):
    """
    将识别到的 item_name 写入 state 和每个 Chunk 的元数据中
    目的：后续 Milvus 入库时每个 Chunk 携带 item_name，支持按商品过滤检索
    :param state: 流程状态对象
    :param chunks: 切片列表（就地修改）
    :param item_name: 步骤3 识别的商品名称
    """
    state["item_name"] = item_name
    for chunk in chunks:
        chunk["item_name"] = item_name
    state["chunks"] = chunks
    logger.info(f"步骤4：商品名称回填完成，共为{len(chunks)}个切片添加item_name字段，值为：{item_name}")


# ============================================================
# 步骤 5：为商品名称生成 BGE-M3 双向量
# ============================================================

def step_5_generate_vectors(item_name: str) -> Tuple[Any, Any]:
    """
    为识别出的商品名称生成稠密 + 稀疏双向量
    稠密向量（1024维）：捕捉语义（如 "万用表" ≈ "多用表"）
    稀疏向量（变长）：捕捉关键词（如 "Fluke"、"17B+"）
    :param item_name: 商品名称
    :return: (dense_vector, sparse_vector)，异常时返回 (None, None)
    """
    logger.info(f"开始执行步骤5：为商品名称[{item_name}]生成BGE-M3双向量")

    if not item_name:
        logger.warning("商品名称为空，跳过向量生成，返回空向量")
        return None, None

    try:
        vector_result = generate_embeddings([item_name])

        if vector_result and "dense" in vector_result and "sparse" in vector_result:
            dense_vector = vector_result["dense"][0]
            sparse_vector = vector_result["sparse"][0]
            logger.info("步骤5：BGE-M3稠密+稀疏向量生成成功")
        else:
            logger.warning("步骤5：向量生成工具返回空结果")
            dense_vector, sparse_vector = None, None

    except Exception as e:
        logger.error(f"步骤5：向量生成失败，原因：{str(e)}", exc_info=True)
        dense_vector, sparse_vector = None, None

    return dense_vector, sparse_vector


# ============================================================
# 步骤 6：将商品名称及向量持久化到 Milvus
# ============================================================

def step_6_save_to_milvus(
    state: ImportGraphState,
    file_title: str,
    item_name: str,
    dense_vector,
    sparse_vector
):
    """
    将商品名称、文件标题、双向量写入 Milvus 的 item_names 集合
    完整流程：创建集合（不存在时）→ 幂等删除旧数据 → 插入新数据 → 加载集合
    :param state: 流程状态对象
    :param file_title: 文件标题
    :param item_name: 商品名称
    :param dense_vector: 稠密向量（1024维列表）
    :param sparse_vector: 稀疏向量（字典格式）
    """
    milvus_uri = os.environ.get("MILVUS_URL")
    collection_name = os.environ.get("ITEM_NAME_COLLECTION")

    if not all([milvus_uri, collection_name]):
        logger.warning("Milvus配置缺失（MILVUS_URL/ITEM_NAME_COLLECTION），跳过数据保存")
        return

    logger.info(f"开始执行步骤6：将商品名称[{item_name}]保存到Milvus集合[{collection_name}]")

    try:
        client = get_milvus_client()
        if not client:
            logger.error("无法获取Milvus客户端（连接失败），跳过数据保存")
            return

        # 集合初始化：不存在则创建 Schema + 索引
        if not client.has_collection(collection_name=collection_name):
            logger.info(f"Milvus集合[{collection_name}]不存在，开始创建Schema和索引")

            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

            # 自增主键
            schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
            # 文件标题
            schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
            # 商品名称（去重依据）
            schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
            # 稠密向量（BGE-M3 固定 1024 维）
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
            # 稀疏向量（变长）
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

            # 构建索引参数
            index_params = client.prepare_index_params()

            # 稠密向量索引：HNSW + COSINE（搜索快、精度高）
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200}
            )

            # 稀疏向量索引：倒排索引 + IP（专业级稀疏检索）
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE", "quantization": "none"}
            )

            client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
            logger.info(f"Milvus集合[{collection_name}]创建成功，包含Schema和向量索引")

        # 幂等性处理：删除同名商品数据，避免重复存储
        clean_item_name = (item_name or "").strip()
        if clean_item_name:
            client.load_collection(collection_name=collection_name)
            safe_item_name = escape_milvus_string(clean_item_name)
            filter_expr = f'item_name=="{safe_item_name}"'
            client.delete(collection_name=collection_name, filter=filter_expr)
            logger.info(f"Milvus幂等性处理完成，已删除集合中[{clean_item_name}]的历史数据")

        # 构造插入数据
        data = {
            "file_title": file_title,
            "item_name": item_name
        }
        if dense_vector is not None:
            data["dense_vector"] = dense_vector
        if sparse_vector is not None:
            data["sparse_vector"] = sparse_vector

        client.insert(collection_name=collection_name, data=[data])
        client.load_collection(collection_name=collection_name)

        state["item_name"] = item_name
        logger.info(f"步骤6：商品名称[{item_name}]成功存入Milvus集合[{collection_name}]")

    except Exception as e:
        logger.error(f"步骤6：数据存入Milvus失败，原因：{str(e)}", exc_info=True)


# ============================================================
# 节点主入口：node_item_name_recognition
# ============================================================

def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    商品主体名称识别节点 — 六步法完成识别 + 向量化 + 入库
    节点定位：
    - 上游：node_document_split（state 中有 file_title、chunks）
    - 下游：node_bge_embedding（需要 chunks 中的 item_name 字段）
    核心流程：
    1. 提取输入：从 state 取 file_title + chunks
    2. 构建上下文：取前 K 个切片拼成 LLM 识别的素材
    3. LLM 识别：调用大模型识别商品名，三层兜底
    4. 回填数据：item_name 写入 state 和每个 Chunk
    5. 生成向量：BGE-M3 对商品名生成稠密+稀疏双向量
    6. 存入 Milvus：写入 item_names 集合（幂等删除+插入）
    :param state: 导入流程全局状态对象
    :return: 更新后的状态对象（state['item_name']、state['chunks'] 已填充）
    """
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{node_name}] 开始执行商品名称识别节点")
    add_running_task(state.get("task_id", ""), node_name)

    try:
        # ========================================
        # 步骤 1：提取并校验输入数据
        # ========================================
        file_title, chunks = step_1_get_inputs(state)
        if not chunks:
            logger.warning(f">>> [{node_name}] 无有效切片数据，跳过识别")
            return state

        # ========================================
        # 步骤 2：构建 LLM 识别上下文
        # ========================================
        context = step_2_build_context(chunks)

        # ========================================
        # 步骤 3：调用 LLM 识别商品名称
        # ========================================
        item_name = step_3_call_llm(file_title, context)

        # ========================================
        # 步骤 4：回填数据到 state 和所有切片
        # ========================================
        step_4_update_chunks(state, chunks, item_name)

        # ========================================
        # 步骤 5：生成 BGE-M3 双向量
        # ========================================
        dense_vector, sparse_vector = step_5_generate_vectors(item_name)

        # ========================================
        # 步骤 6：存入 Milvus 向量数据库
        # ========================================
        step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        logger.info(f">>> [{node_name}] 节点执行完成，识别结果：{item_name}")

    except Exception as e:
        logger.error(f">>> [{node_name}] 节点执行失败：{str(e)}", exc_info=True)
        state["item_name"] = "未知商品"

    finally:
        add_done_task(state["task_id"], node_name)

    return state


# ============================================================
# 单元测试入口
# ============================================================

if __name__ == "__main__":
    """
    本地测试：模拟上游节点产出，独立测试商品名称识别全流程
    测试前准备：
    - .env 已配置 MILVUS_URL / ITEM_NAME_COLLECTION / LLM
    - Milvus 服务已启动
    """
    from app.utils.path_util import PROJECT_ROOT

    logger.info("=== 开始执行商品名称识别节点本地测试 ===")

    # 构造模拟 state（模拟上游 node_document_split 产出）
    test_state: ImportGraphState = {
        "task_id": "test_item_001",
        "file_title": "hak180产品安全手册",
        "chunks": [
            {
                "title": "# 安全须知",
                "content": "hak180 是一款工业级热风枪，使用前请仔细阅读本安全手册。本产品适用于电子维修、热缩管加工等场景。",
                "file_title": "hak180产品安全手册",
            },
            {
                "title": "## 电气安全",
                "content": "请确保使用额定电压 220V 的电源插座。不要在潮湿环境下使用本设备。电源线损坏时请立即更换。",
                "file_title": "hak180产品安全手册",
            },
            {
                "title": "## 操作规范",
                "content": "使用时请佩戴防护手套。热风枪工作时喷嘴温度可达 500°C，请勿触碰。使用完毕后请放置于专用支架冷却。",
                "file_title": "hak180产品安全手册",
            },
            {
                "title": "## 维护保养",
                "content": "定期清理进风口滤网，避免灰尘堵塞导致过热。长期不使用时请存放于干燥处。",
                "file_title": "hak180产品安全手册",
            },
        ],
    }

    result = node_item_name_recognition(test_state)
    logger.info(f"测试完成 - item_name={result.get('item_name')}")
    logger.info(f"测试完成 - chunks数量={len(result.get('chunks', []))}")
