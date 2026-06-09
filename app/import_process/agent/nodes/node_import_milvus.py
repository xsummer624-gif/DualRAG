import os
import sys
from typing import List, Dict, Any

from pymilvus import DataType

from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task, add_done_task
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

# 从配置文件读取集合名称
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection


# ==========================================
# 集合创建（Schema + 双索引）
# ==========================================

def create_collection(client, collection_name: str, vector_dimension: int):
    """
    Milvus 集合 + 双向量索引自动创建
    字段：chunk_id(自增主键) / content / title / parent_title / part /
          file_title / item_name / dense_vector(1024维) / sparse_vector(变长)
    索引：稠密 HNSW + COSINE / 稀疏 SPARSE_INVERTED_INDEX + IP
    """
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

    # 自增主键
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    # 业务字段
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
    # 向量字段
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # 构建索引
    index_params = client.prepare_index_params()

    # 稠密向量索引：HNSW + COSINE
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )

    # 稀疏向量索引：倒排索引 + IP
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "quantization": "none"}
    )

    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info(f"Milvus集合创建成功：{collection_name}，向量维度：{vector_dimension}")


# ==========================================
# 步骤 1：校验输入数据
# ==========================================

def step_1_check_input(state: ImportGraphState) -> tuple:
    """
    校验 chunks 有效性：非空列表、包含 dense_vector 字段
    :return: (chunks列表, 向量维度)
    """
    chunks_data = state.get("chunks")
    if not chunks_data:
        raise ValueError("错误: chunks为空，无法执行Milvus入库")
    if not isinstance(chunks_data, list) or len(chunks_data) == 0:
        raise ValueError("错误: chunks数据格式不正确，必须为非空列表")

    first_chunk = chunks_data[0]
    if "dense_vector" not in first_chunk:
        raise ValueError("错误: 数据中缺失dense_vector字段，上游向量化节点可能执行失败")

    vector_dimension = len(first_chunk["dense_vector"])
    item_name = first_chunk.get("item_name", "未知商品名")
    logger.info(f"Milvus入库校验通过，切片数：{len(chunks_data)} | 向量维度：{vector_dimension} | 商品名：{item_name}")

    return chunks_data, vector_dimension


# ==========================================
# 步骤 2：准备集合（连接 + 自动建表）
# ==========================================

def step_2_prepare_collection(vector_dimension: int):
    """
    连接 Milvus，集合不存在则自动创建
    """
    logger.info(f"准备Milvus环境，目标集合：{CHUNKS_COLLECTION_NAME}")
    client = get_milvus_client()
    if client is None:
        raise ValueError("Milvus连接失败：get_milvus_client()返回空")
    if not CHUNKS_COLLECTION_NAME:
        raise ValueError("未配置CHUNKS_COLLECTION集合名称")

    if not client.has_collection(collection_name=CHUNKS_COLLECTION_NAME):
        logger.info(f"集合{CHUNKS_COLLECTION_NAME}不存在，自动创建")
        create_collection(client, CHUNKS_COLLECTION_NAME, vector_dimension)
    else:
        logger.info(f"集合{CHUNKS_COLLECTION_NAME}已存在，直接复用")

    return client


# ==========================================
# 步骤 3：幂等性清理旧数据
# ==========================================

def _clear_chunks_by_item_name(client, collection_name: str, item_name: str):
    """根据 item_name 删除集合中的旧数据"""
    i_name = (item_name or "").strip()
    if not i_name or not collection_name:
        return
    if not client.has_collection(collection_name=collection_name):
        return

    try:
        safe_item_name = escape_milvus_string(i_name)
        filter_expr = f'item_name == "{safe_item_name}"'
        client.delete(collection_name=collection_name, filter=filter_expr)
        logger.info(f"Milvus幂等清理完成：已删除item_name={i_name}的旧数据")
    except Exception as e:
        logger.error(f"Milvus幂等清理失败：item_name={i_name} | {str(e)}", exc_info=True)
        raise


def step_3_clean_old_data(client, chunks_data: List[Dict[str, Any]]):
    """
    提取所有 item_name 并去重，逐个清理旧数据
    """
    item_names = sorted({
        name
        for x in (chunks_data or [])
        if (name := str(x.get("item_name", "")).strip())
    })

    if not item_names:
        logger.warning("切片中无有效item_name，跳过幂等清理")
        return

    for name in item_names:
        _clear_chunks_by_item_name(client, CHUNKS_COLLECTION_NAME, name)


# ==========================================
# 步骤 4：批量插入 + chunk_id 回填
# ==========================================

def step_4_insert_data(client, chunks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    批量插入切片数据到 Milvus，将自增 chunk_id 回填到每个切片
    """
    # 移除手动 chunk_id（auto_id=True 时 Milvus 自动生成）
    data_to_insert = []
    for item in chunks_data:
        item_copy = item.copy()
        item_copy.pop("chunk_id", None)
        data_to_insert.append(item_copy)

    logger.info(f"开始批量插入{len(data_to_insert)}条数据到Milvus")
    insert_result = client.insert(collection_name=CHUNKS_COLLECTION_NAME, data=data_to_insert)
    insert_count = insert_result.get("insert_count", 0)
    logger.info(f"Milvus插入完成：成功{insert_count}条")

    # 回填 chunk_id
    inserted_ids = insert_result.get("ids", [])
    if inserted_ids and len(inserted_ids) == len(chunks_data):
        for idx, item in enumerate(chunks_data):
            item["chunk_id"] = str(inserted_ids[idx])
        logger.info("chunk_id回填完成")
    else:
        logger.warning(f"chunk_id回填数量不一致：生成{len(inserted_ids)} vs 切片{len(chunks_data)}")

    return chunks_data


# ==========================================
# 节点主入口
# ==========================================

def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    Milvus 入库节点 — 将带向量的 Chunk 批量写入 Milvus
    节点定位：
    - 上游：node_bge_embedding（chunks 中有 dense_vector / sparse_vector）
    - 下游：END（导入流水线终点）
    核心流程：
    1. 校验输入（chunks + 向量字段完整性）
    2. 准备集合（自动建表 + 双索引）
    3. 幂等清理（按 item_name 删旧数据）
    4. 批量插入 + chunk_id 回填
    :param state: 导入流程全局状态对象
    :return: 更新后的状态对象（chunks 回填了 chunk_id）
    """
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> [{current_node}] 开始执行 Milvus 入库节点")
    add_running_task(state.get("task_id", ""), current_node)

    try:
        # 步骤 1：校验输入
        chunks_data, vector_dim = step_1_check_input(state)

        # 步骤 2：准备集合
        client = step_2_prepare_collection(vector_dim)

        # 步骤 3：幂等清理
        step_3_clean_old_data(client, chunks_data)

        # 步骤 4：批量插入 + chunk_id 回填
        updated_chunks = step_4_insert_data(client, chunks_data)
        state["chunks"] = updated_chunks

        logger.info(f">>> [{current_node}] 入库完成，共写入 {len(updated_chunks)} 条数据")

    except Exception as e:
        logger.error(f">>> [{current_node}] 节点执行失败：{str(e)}", exc_info=True)
        raise

    finally:
        add_done_task(state.get("task_id", ""), current_node)

    return state


# ==========================================
# 单元测试
# ==========================================

if __name__ == "__main__":
    """
    本地测试：模拟上游向量化节点输出，验证 Milvus 入库全流程
    """
    logger.info("=== Milvus入库节点本地单元测试 ===")

    dim = 1024
    test_state: ImportGraphState = {
        "task_id": "test_milvus_001",
        "chunks": [
            {
                "content": "hak180 是一款工业级热风枪。",
                "title": "# 安全须知",
                "item_name": "测试商品_Milvus",
                "parent_title": "# 安全须知",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8},
            }
        ],
    }

    if not os.getenv("MILVUS_URL"):
        logger.error("未设置 MILVUS_URL，请检查 .env 配置")
    elif not os.getenv("CHUNKS_COLLECTION"):
        logger.error("未设置 CHUNKS_COLLECTION")
    else:
        try:
            result_state = node_import_milvus(test_state)
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                logger.info(f"✅ 测试通过，chunk_id={chunks[0]['chunk_id']}")
            else:
                logger.error("❌ 未获取到 chunk_id")
        except Exception as e:
            logger.error(f"❌ 测试失败: {e}")
