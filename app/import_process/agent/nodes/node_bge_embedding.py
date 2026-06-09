import sys
from typing import Any, List, Dict

from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


# ==========================================
# 步骤 1：校验输入数据
# ==========================================

def step_1_validate_input(state: ImportGraphState) -> List[Dict[str, Any]]:
    """
    校验 state 中的 chunks 是否有效
    :param state: 流程全局状态对象
    :return: 校验通过的 chunks 列表
    :raise ValueError: chunks 为空或非列表时抛出
    """
    texts_to_embed = state.get("chunks")
    if not isinstance(texts_to_embed, list) or not texts_to_embed:
        logger.error("向量化输入校验失败：chunks字段为空或非有效列表")
        raise ValueError("错误: 无有效文本切片数据，无法执行向量化处理")

    logger.info(f"向量化输入校验通过，待处理文本切片数量：{len(texts_to_embed)}")
    return texts_to_embed


# ==========================================
# 步骤 2：初始化 BGE-M3 模型（单例）
# ==========================================

def step_2_init_model():
    """
    获取 BGE-M3 单例模型实例，模型全局只加载一次
    :return: BGEM3EmbeddingFunction 实例
    """
    try:
        ef = get_bge_m3_ef()
        if ef is None:
            raise ValueError("BGE-M3模型实例为None，模型加载失败")
        logger.info("BGE-M3模型实例初始化成功（单例模式）")
        return ef
    except Exception as e:
        error_msg = f"BGE-M3模型初始化失败：{e}，请检查模型路径/环境变量配置是否正确"
        logger.error(error_msg)
        raise ValueError(error_msg)


# ==========================================
# 步骤 3：批量生成稠密 + 稀疏双向量
# ==========================================

def step_3_generate_embeddings(
    texts_to_embed: List[Dict[str, Any]],
    bge_m3_ef: Any
) -> List[Dict[str, Any]]:
    """
    分批次为每个 Chunk 生成 BGE-M3 双向量，写入 dense_vector 和 sparse_vector 字段
    核心设计：item_name 前置拼接，强化核心特征，提升检索精度
    :param texts_to_embed: 校验通过的切片列表
    :param bge_m3_ef: BGE-M3 模型实例
    :return: 带向量字段的切片列表
    """
    output_data = []
    batch_size = 5  # 每批处理 5 条，平衡显存占用和处理效率
    total = len(texts_to_embed)

    for i in range(0, total, batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        start_idx, end_idx = i + 1, min(i + len(batch_texts), total)

        try:
            # 构造模型输入：item_name（商品名）+ content（正文）
            # "核心词前置"原则——Embedding 模型对前 128 token 注意力最集中
            input_texts = []
            for doc in batch_texts:
                item_name = doc.get("item_name", "")
                content = doc.get("content", "")
                text = f"商品：{item_name}，介绍：{content}" if item_name else content
                input_texts.append(text)

            # 调用封装函数生成批量双向量
            docs_embeddings = generate_embeddings(input_texts)
            if not docs_embeddings:
                logger.warning(f"第{start_idx}-{end_idx}条切片：向量生成返回空，保留原数据")
                output_data.extend(batch_texts)
                continue

            # 为每个切片绑定对应向量
            for j, doc in enumerate(batch_texts):
                item = doc.copy()
                item["dense_vector"] = docs_embeddings["dense"][j]
                item["sparse_vector"] = docs_embeddings["sparse"][j]
                output_data.append(item)

            logger.info(f"第{start_idx}-{end_idx}条切片：双向量生成成功")

        except Exception as e:
            logger.error(f"第{start_idx}-{end_idx}条切片：向量生成失败 | {str(e)}", exc_info=True)
            output_data.extend(batch_texts)

    return output_data


# ==========================================
# 节点主入口
# ==========================================

def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    BGE-M3 向量化节点 — 将每个 Chunk 的文本转为稠密 + 稀疏双向量
    节点定位：
    - 上游：node_item_name_recognition（chunks 中有 item_name、content）
    - 下游：node_import_milvus（需要 dense_vector + sparse_vector 写入 Milvus）
    核心流程：
    1. 校验输入 chunks
    2. 加载 BGE-M3 模型（单例）
    3. 分批生成双向量并绑定到每个 Chunk
    4. 更新 state["chunks"]
    :param state: 导入流程全局状态对象
    :return: 更新后的状态对象（chunks 多了 dense_vector / sparse_vector）
    """
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> [{current_node}] 开始执行 BGE-M3 向量化节点")
    add_running_task(state.get("task_id", ""), current_node)

    try:
        # 步骤 1：校验输入
        texts_to_embed = step_1_validate_input(state)

        # 步骤 2：初始化模型
        bge_m3_ef = step_2_init_model()

        # 步骤 3：批量生成双向量
        output_data = step_3_generate_embeddings(texts_to_embed, bge_m3_ef)

        # 步骤 4：更新 state
        state["chunks"] = output_data
        logger.info(f">>> [{current_node}] 向量化完成，共处理 {len(output_data)} 条切片")

    except Exception as e:
        logger.error(f">>> [{current_node}] 节点执行失败：{str(e)}", exc_info=True)

    finally:
        add_done_task(state.get("task_id", ""), current_node)

    return state


# ==========================================
# 单元测试
# ==========================================

if __name__ == "__main__":
    """
    本地测试：模拟上游节点输出，验证向量化全流程
    """
    logger.info("=== BGE-M3向量化节点本地单元测试 ===")

    test_state: ImportGraphState = {
        "task_id": "test_embedding_001",
        "chunks": [
            {
                "content": "hak180 是一款工业级热风枪，适用于电子维修场景。",
                "title": "# 安全须知",
                "item_name": "hak180工业级热风枪",
                "file_title": "hak180产品安全手册",
            },
            {
                "content": "使用时请佩戴防护手套，喷嘴温度可达500°C。",
                "title": "## 操作规范",
                "item_name": "hak180工业级热风枪",
                "file_title": "hak180产品安全手册",
            },
        ],
    }

    result_state = node_bge_embedding(test_state)
    result_chunks = result_state.get("chunks", [])

    for idx, chunk in enumerate(result_chunks):
        has_dense = "dense_vector" in chunk
        has_sparse = "sparse_vector" in chunk
        logger.info(f"切片{idx+1}：稠密{'✅' if has_dense else '❌'} | 稀疏{'✅' if has_sparse else '❌'}")
