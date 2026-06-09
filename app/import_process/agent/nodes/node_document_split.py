import re
import json
import os
import sys
# 统一类型注解
from typing import List, Dict, Any, Tuple
# LangChain 文本分割器：递归切分，从粗到细保留语义
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 项目内部工具/状态/日志导入
from app.utils.task_utils import add_running_task, add_done_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger

# ============================================================
# 配置参数
# ============================================================

# 单个 Chunk 最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000

# 短 Chunk 合并阈值：同父标题的短 Chunk 会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500


# ============================================================
# 步骤 1：获取并标准化输入数据
# ============================================================

def step_1_get_inputs(state: ImportGraphState) -> Tuple[Any, str, int]:
    """
    从状态字典中提取 MD 内容 / 文件标题 / 最大长度，做基础标准化
    上游节点 node_md_img 已将处理后的 md_content、file_title 写入 state
    :param state: 项目状态字典（ImportGraphState）
    :return: 三元组 (标准化后的 MD 内容, 文件标题, Chunk 最大长度)
             内容为空时返回 (None, None, None)
    """
    # 1. 提取 MD 内容 — 这是上游节点处理后的完整文本（图片已换为 MinIO URL）
    content = state.get("md_content")
    if not content:
        logger.warning("状态字典中无有效MD内容，终止文档切分")
        return None, None, None

    # 2. 统一换行符：消除 Windows(\r\n) 和 Linux(\n) 的系统差异
    #    例如 "# HL3070说明书\r\n## 产品概述\n" → "# HL3070说明书\n## 产品概述\n"
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # 3. 提取文件标题：有则用，无则兜底（node_entry 已设置）
    file_title = state.get("file_title", "Unknown File")

    # 4. 确定 Chunk 最大长度（后续可扩展为从 state 读取用户自定义配置）
    max_len = DEFAULT_MAX_CONTENT_LENGTH

    logger.info(f"步骤1：输入数据加载完成，文件标题：{file_title}，最大Chunk长度：{max_len}")
    return content, file_title, max_len


# ============================================================
# 步骤 2：按 Markdown 标题初次切分
# 核心逻辑：逐行遍历 MD，遇到标题行就「结算」上一个章节并开始新章节
# 同时跳过代码块内的 # 号，避免误判为标题
# ============================================================

def step_2_split_by_titles(content: str, file_title: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    基于 Markdown 标题语法（# / ## / ### ...）进行第一轮粗粒度切分
    每个章节 = 标题 + 标题下所有内容行，保证同一知识点的语义连贯性
    :param content: 标准化后的 MD 完整内容（已统一换行符）
    :param file_title: 所属文件标题，用于标记章节归属
    :return: 三元组 (切分后的章节列表, 有效标题数量, MD 原始总行数)
    """
    # 正则匹配 Markdown 1-6 级标题
    # ^\s*       → 行首允许 0/多个空格/Tab（兼容缩进）
    # #{1,6}     → 匹配 1-6 个 # 号（对应 MD 的 1-6 级标题）
    # \s+        → # 后必须有至少 1 个空格（区分 # 是标题还是注释）
    # .+         → 标题文字至少 1 个字符（避免空标题行）
    title_pattern = r'^\s*#{1,6}\s+.+'

    # 将 MD 内容按换行符拆分为行列表，逐行处理
    lines = content.split("\n")
    sections = []          # 最终切分的章节列表
    current_title = ""     # 当前正在处理的章节标题
    current_lines = []     # 当前章节的行缓存（标题 + 正文）
    title_count = 0        # 有效标题计数器（排除代码块内伪标题）
    in_code_block = False  # 代码块状态标记（``` 或 ~~~ 之间）

    def _flush_section():
        """内部辅助函数：结算当前缓存的章节，写入 sections 列表"""
        if not current_lines:
            return
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines),  # 恢复换行连接各行
            "file_title": file_title,
        })

    # 逐行遍历 MD 文本
    for line in lines:
        stripped_line = line.strip()

        # 识别代码块边界（``` 或 ~~~）：进入/退出时翻转状态
        # 这一步至关重要：代码块内常有 # 注释，不跳过会导致伪标题污染章节切分
        if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        # 判断是否为有效标题：非代码块内 + 匹配标题正则
        is_valid_title = (not in_code_block) and re.match(title_pattern, line)

        if is_valid_title:
            # 遇到新标题 → 先结算上一个章节，再初始化新章节
            _flush_section()
            current_title = line.strip()            # 清理标题首尾空格
            current_lines = [current_title]          # 新章节从标题行开始
            title_count += 1
            logger.debug(f"识别到MD标题：{current_title}")
        else:
            # 普通正文行 → 追加到当前章节的行缓存
            current_lines.append(line)

    # 循环结束：结算最后一个缓存的章节
    _flush_section()

    logger.info(f"步骤2：MD标题切分完成，识别到{title_count}个有效标题，原始文本共{len(lines)}行")
    return sections, title_count, len(lines)


# ============================================================
# 步骤 3：无标题兜底处理
# 处理那些完全没有 Markdown 标题的纯文本文件
# ============================================================

def step_3_handle_no_title(
    content: str,
    sections: List[Dict[str, Any]],
    title_count: int,
    file_title: str
) -> List[Dict[str, Any]]:
    """
    无标题场景兜底：若 MD 未识别到任何标题，将全文封装为单个「无标题」章节
    这样后续的精细化切分逻辑无需特殊处理，保证数据格式统一
    :param content: 标准化后的 MD 完整内容
    :param sections: 步骤 2 切分后的章节列表
    :param title_count: 步骤 2 识别的有效标题数量
    :param file_title: 所属文件标题
    :return: 兜底后的章节列表（有标题直接返回原列表，无标题返回单章节列表）
    """
    if title_count == 0:
        # 无标题：替换为单章节结构，标题统一为"无标题"
        logger.warning(f"步骤3：未识别到任何MD标题，将全文作为单个章节处理，文件：{file_title}")
        return [{"title": "无标题", "content": content, "file_title": file_title}]

    # 有标题：直接返回步骤 2 的结果
    logger.debug(f"步骤3：检测到{title_count}个有效标题，无需兜底处理")
    return sections


# ============================================================
# 辅助函数：超长章节二次切分
# ============================================================

def _split_long_section(
    section: Dict[str, Any],
    max_length: int = DEFAULT_MAX_CONTENT_LENGTH
) -> List[Dict[str, Any]]:
    """
    超长章节二次切分（使用 LangChain RecursiveCharacterTextSplitter）
    切分策略（从粗到细）：
    1. 先按空行（段落）切 → 2. 再按换行切 → 3. 再按中文标点切
    → 4. 再按英文标点切 → 5. 最后硬按空格切
    这样能最大限度保留语义完整性，避免在句子中间截断
    :param section: 原始章节字典，必须包含 content 键
    :param max_length: 单个 Chunk 最大字符长度
    :return: 切分后的子章节列表，每个子章节保留父级标题等元信息
    """
    content = section.get("content", "") or ""

    # 长度未超限 → 无需切分，直接返回原章节（列表格式保持统一）
    if len(content) <= max_length:
        return [section]

    # 统一换行符（防止上游混合换行影响切分结果）
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # 提取章节标题，用于拼子 Chunk 的前缀
    title = section.get("title", "") or ""
    # 标题前缀：带空行分隔标题和正文
    prefix = f"{title}\n\n" if title else ""

    # 计算正文可用长度 = 总限制 - 标题前缀长度
    # 避免标题本身占满 Chunk 额度导致正文空间为零
    available_len = max_length - len(prefix)

    # 极端情况：标题本身长度超过阈值，无法切分
    if available_len <= 0:
        logger.warning(f"章节标题过长，无法切分：{title[:20]}...")
        return [section]

    # 清理正文中重复的标题行
    # MinerU 解析时标题可能在正文第一行重复出现，去掉它避免内容冗余
    body = content
    if title and body.lstrip().startswith(title):
        body = body[body.find(title) + len(title):].lstrip()

    # 初始化 LangChain 递归分割器
    # separators 优先级从高到低：先尝试用大语义单元（段落）切分，
    # 失败则逐级降级到更细粒度的分隔符，最后才硬拆
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_len,      # 正文可用长度
        chunk_overlap=0,               # 不设重叠：按标题切分语义已完整
        separators=[
            "\n\n",                     # 1级：段落（空行）
            "\n",                       # 2级：换行
            "。", "！", "？", "；",     # 3级：中文标点
            ".", "!", "?", ";",        # 4级：英文标点
            " "                         # 5级：空格（最后手段）
        ],
    )

    # 切分正文并组装子章节（带完整元信息）
    sub_sections = []
    for idx, chunk in enumerate(splitter.split_text(body), start=1):
        text = chunk.strip()
        if not text:
            continue  # 跳过空内容

        # 组装子 Chunk：标题前缀 + 切分后的正文
        full_text = (prefix + text).strip()

        sub_sections.append({
            "title": f"{title}-{idx}" if title else f"chunk-{idx}",  # 带序号的子标题
            "content": full_text,                                      # 完整内容
            "parent_title": title,                                     # 所属父标题
            "part": idx,                                               # 子序号
            "file_title": section.get("file_title"),                   # 所属文件
        })

    logger.debug(f"超长章节切分完成：{title} → 生成{len(sub_sections)}个子Chunk")
    return sub_sections


# ============================================================
# 辅助函数：过短章节合并
# ============================================================

def _merge_short_sections(
    sections: List[Dict[str, Any]],
    min_length: int = MIN_CONTENT_LENGTH
) -> List[Dict[str, Any]]:
    """
    过短章节合并：将同父标题下相邻的短 Chunk 合并，减少碎片化
    核心规则：只合并「同父标题」+「当前块长度不足阈值」的相邻 Chunk
    跨章节绝不合并（保证不同知识点的独立性）
    :param sections: 待合并的 Chunk 列表
    :param min_length: 最小长度阈值，低于此值触发合并
    :return: 合并后的 Chunk 列表
    """
    if not sections:
        return []

    merged = []             # 最终结果
    current_chunk = None    # 迭代累加器：当前待合并的 Chunk

    for sec in sections:
        # 第一个 Chunk：直接作为累加器起点
        if current_chunk is None:
            current_chunk = sec
            continue

        # 合并条件判断：
        # ① 当前块长度不足阈值（碎片化）
        # ② 与下一块属于同一个父标题（同章节，语义连贯）
        is_current_short = len(current_chunk["content"]) < min_length
        is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

        if is_current_short and is_same_parent:
            # 满足合并条件：拼接两块的内容
            parent_title = sec.get("parent_title", "")
            next_content = sec["content"]

            # 清理下一块开头重复的父标题，避免内容冗余
            if parent_title and next_content.startswith(parent_title):
                next_content = next_content[len(parent_title):].lstrip()

            # 空行分隔拼接
            current_chunk["content"] += "\n\n" + next_content

            # 更新序号为最新
            if "part" in sec:
                current_chunk["part"] = sec["part"]

            logger.debug(f"合并短Chunk：{current_chunk.get('parent_title')} → 累计长度{len(current_chunk['content'])}")
        else:
            # 不满足合并条件：累加器结算，切换为新块
            merged.append(current_chunk)
            current_chunk = sec

    # 循环结束：结算最后一个累加器
    if current_chunk is not None:
        merged.append(current_chunk)

    logger.debug(f"短Chunk合并完成：原{len(sections)}个 → 合并后{len(merged)}个")
    return merged


# ============================================================
# 步骤 4：Chunk 精细化处理（长切短合 + 父标题兜底）
# ============================================================

def step_4_refine_chunks(sections: List[Dict[str, Any]], max_len: int) -> List[Dict[str, Any]]:
    """
    Chunk 精细化处理 — 三阶段流水线：
    1. 切分超长章节（_split_long_section）
    2. 合并过短章节（_merge_short_sections）
    3. 父标题兜底（适配 Milvus 向量库必填字段）
    :param sections: 步骤 3 处理后的章节列表
    :param max_len: 单个 Chunk 最大字符长度
    :return: 长度适中、语义完整、低碎片化的最终 Chunk 列表
    """
    # 边界处理：max_len 无效则跳过精细化
    if not max_len or max_len <= 0:
        logger.warning(f"步骤4：Chunk最大长度配置无效（{max_len}），跳过精细化处理")
        return sections

    # 阶段 1：切分超长章节
    # extend() 将切分结果平铺展开（每个超长章节 → 多个子 Chunk）
    refined_split = []
    for sec in sections:
        refined_split.extend(_split_long_section(sec, max_len))
    logger.info(f"步骤4-1：超长章节切分完成，共生成{len(refined_split)}个初始子Chunk")

    # 阶段 2：合并过短章节（减少碎片化，提升检索效果）
    final_sections = _merge_short_sections(refined_split)
    logger.info(f"步骤4-2：过短章节合并完成，最终得到{len(final_sections)}个Chunk")

    # 阶段 3：父标题兜底
    # Milvus 向量库 schema 中 parent_title 为必填字段，缺失会导致写入失败
    for sec in final_sections:
        if not isinstance(sec, dict):
            continue

        # part 字段兜底：序号缺失则补 0
        if "part" not in sec:
            sec["part"] = 0

        # parent_title 兜底：无父标题则用自身标题，自身也无则填空字符串
        if not sec.get("parent_title"):
            sec["parent_title"] = sec.get("title") or ""

    logger.debug("步骤4-3：父标题兜底完成，所有Chunk均包含parent_title字段")
    return final_sections


# ============================================================
# 步骤 5：打印切分统计信息
# ============================================================

def step_5_print_stats(lines_count: int, sections: List[Dict[str, Any]]) -> None:
    """
    输出文档切分统计信息（纯日志，无副作用）
    便于开发调试和线上监控切分效果
    :param lines_count: MD 原始文本总行数
    :param sections: 最终处理后的 Chunk 列表
    """
    chunk_num = len(sections)
    logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
    logger.info(f"MD原始文本总行数：{lines_count}")
    logger.info(f"最终生成Chunk数量：{chunk_num}")

    if sections:
        first_title = sections[0].get("title", "无标题")
        logger.info(f"首个Chunk标题预览：{first_title}")

        # 额外输出每个 Chunk 的长度分布概览
        lengths = [len(s.get("content", "")) for s in sections]
        logger.info(f"Chunk长度范围：{min(lengths)} ~ {max(lengths)} 字符")
        logger.info(f"Chunk平均长度：{sum(lengths) // len(lengths)} 字符")

    logger.info("-" * 110)


# ============================================================
# 步骤 6：Chunk 结果 JSON 备份 + 状态更新
# ============================================================

def step_6_backup(state: ImportGraphState, sections: List[Dict[str, Any]]) -> None:
    """
    将最终 Chunk 列表备份为本地 chunks.json 文件
    目的：
    - 便于问题排查和数据复查（直接打开 JSON 就能看到切分结果）
    - 为后续节点提供可追溯的数据快照
    :param state: 项目状态字典，需包含 local_dir（备份目录路径）
    :param sections: 最终处理后的 Chunk 列表
    """
    local_dir = state.get("local_dir")
    if not local_dir:
        logger.warning("步骤6：未配置备份目录（local_dir），跳过Chunk结果备份")
        return

    try:
        # 确保备份目录存在（exist_ok=True 幂等）
        os.makedirs(local_dir, exist_ok=True)

        # 拼接备份文件路径：local_dir/chunks.json
        backup_path = os.path.join(local_dir, "chunks.json")

        # json.dump 直接将 Python 数据结构序列化写入 JSON 文件
        # ensure_ascii=False → 保留中文不转义（"一级标题" 而非 "\u4e00\u7ea7\u6807\u9898"）
        # indent=2          → 缩进 2 空格，便于人工阅读
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(sections, f, ensure_ascii=False, indent=2)

        logger.info(f"步骤6：Chunk结果备份成功，备份文件路径：{backup_path}")
    except Exception as e:
        # 备份失败仅记录日志，不中断主流程
        logger.error(f"步骤6：Chunk结果备份失败，错误信息：{str(e)}", exc_info=False)


# ============================================================
# 节点主入口：node_document_split
# LangGraph 节点函数，编排 6 个步骤完成文档切分全流程
# ============================================================

def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分核心节点 — 六步法将长 MD 文档转化为适合检索的 Chunk 列表
    节点定位：
    - 上游：node_md_img（已处理图片，md_content 中图片引用为 MinIO URL）
    - 下游：node_item_name_recognition（需要切好的 chunks 来识别商品名）
    核心流程：
    1. 获取输入：提取 md_content + file_title + 切分配置
    2. 标题初切：按 Markdown 标题层级粗粒度切分，跳过代码块内伪标题
    3. 无标题兜底：无标题文档封装为单章节
    4. 精细化处理：超长章节二次切分 + 过短章节合并 + 父标题兜底
    5. 打印统计：输出切分结果概览
    6. 备份与更新：chunks 写入 state + 备份到 chunks.json
    :param state: 导入流程全局状态对象
    :return: 更新后的状态对象（state['chunks'] 已填充）
    """
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{node_name}] 开始执行文档切分节点")
    add_running_task(state["task_id"], node_name)

    try:
        # ========================================
        # 步骤 1：获取并标准化输入数据
        # 从 state 提取 md_content / file_title / max_len
        # ========================================
        content, file_title, max_len = step_1_get_inputs(state)
        if content is None:
            logger.info(f">>> [{node_name}] 无有效MD内容，终止切分")
            return state

        # ========================================
        # 步骤 2：按 Markdown 标题初次切分
        # 逐行遍历 MD → 遇到标题就结算上一章节 → 开始新章节
        # 代码块内 # 号自动跳过，避免伪标题污染
        # ========================================
        sections, title_count, lines_count = step_2_split_by_titles(content, file_title)

        # ========================================
        # 步骤 3：无标题兜底处理
        # 全文档无任何标题 → 封装为单个「无标题」章节
        # ========================================
        sections = step_3_handle_no_title(content, sections, title_count, file_title)

        # ========================================
        # 步骤 4：Chunk 精细化处理（核心：长切短合）
        # ① 超长章节 → LangChain 递归切分为多个子 Chunk
        # ② 过短 Chunk → 同父标题下合并，减少碎片化
        # ③ parent_title 兜底 → 适配 Milvus 必填字段
        # ========================================
        sections = step_4_refine_chunks(sections, max_len)

        # ========================================
        # 步骤 5：输出切分统计信息
        # 纯日志输出：行数 / Chunk 数 / 长度分布
        # ========================================
        step_5_print_stats(lines_count, sections)

        # ========================================
        # 步骤 6：结果写入 state + JSON 备份
        # state['chunks'] 传递给下游节点 node_item_name_recognition
        # chunks.json 存档至 local_dir 供问题排查
        # ========================================
        state["chunks"] = sections
        step_6_backup(state, sections)

        logger.info(f">>> [{node_name}] 节点执行完成，共生成 {len(sections)} 个Chunk")
        return state

    except Exception as e:
        logger.error(f">>> [{node_name}] 节点执行失败：{str(e)}", exc_info=True)
        return state

    finally:
        add_done_task(state["task_id"], node_name)


# ============================================================
# 单元测试入口
# 直接运行 python -m app.import_process.agent.nodes.node_document_split
# ============================================================

if __name__ == "__main__":
    """
    集成测试：联合 node_md_img（图片处理）→ node_document_split（文档切分）
    测试端到端流程：图片处理 → 文档切分，验证两个节点的衔接
    测试条件：
    - .env 已配置 MinIO / LLM 环境
    - output/ 下存在已处理的 MD 文件及其 images/ 子目录
    """
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"集成测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试文件路径（用小文件 hak180 测试，只有 6 张图）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    if not os.path.exists(test_md_path):
        logger.error(f"集成测试 - 测试文件不存在：{test_md_path}")
        logger.info("请先运行 node_md_img 或确保测试文件存在")
    else:
        # 构造测试状态，模拟上游节点传入
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_split_001",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir": os.path.join(PROJECT_ROOT, "output", "hak180产品安全手册"),
        }

        # 阶段 1：运行图片处理节点（Step 3）
        logger.info("\n=== 阶段1：执行图片处理节点 node_md_img ===")
        result_state = node_md_img(test_state)
        logger.info(f"图片处理完成 - md_path={result_state.get('md_path')}")

        # 阶段 2：运行文档切分节点（Step 4）
        logger.info("\n=== 阶段2：执行文档切分节点 node_document_split ===")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"文档切分完成 - 共生成 {len(final_chunks)} 个Chunk")

        if final_chunks:
            # 预览前 3 个 Chunk 的结构
            logger.info("\n=== Chunk 预览（前3个）===")
            for i, chunk in enumerate(final_chunks[:3]):
                logger.info(f"Chunk[{i}] title={chunk.get('title')}, "
                            f"parent_title={chunk.get('parent_title')}, "
                            f"content前50字={chunk.get('content', '')[:50]}...")
