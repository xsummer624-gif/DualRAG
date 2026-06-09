import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO相关依赖：用于步骤4上传和清理旧文件
from minio import Minio
from minio.deleteobjects import DeleteObject

# LangGraph 状态定义：节点函数签名统一使用 ImportGraphState
from app.import_process.agent.state import ImportGraphState
# 任务状态追踪：向前端 SSE 推送进度（开始/结束）
from app.utils.task_utils import add_running_task, add_done_task
# MinIO 客户端单例：避免重复建立连接
from app.clients.minio_utils import get_minio_client
# LLM 客户端工具类：支持多模态模型（qwen-vl等），带缓存机制
from app.lm.lm_utils import get_llm_client
# LangChain 多模态消息构造：用于构造视觉模型的图文混合输入
from langchain.messages import HumanMessage
# LangChain 框架异常捕获：精准捕获框架层错误，提供友好提示
from langchain_core.exceptions import LangChainException
# 项目配置：MinIO 地址/密钥/桶名，LLM 模型名/api
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目统一日志工具
from app.core.logger import logger
# API 速率限制工具：滑动窗口算法，防止触发大模型 API 并发风控
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具：从 prompts/ 目录读取 .prompt 文件并渲染占位符
from app.core.load_prompt import load_prompt

# ============================================================
# MinIO 支持的图片格式集合（小写后缀，统一匹配标准）
# 这些格式均可被 MinIO 对象存储接受，且多模态大模型可识别
# ============================================================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# ============================================================
# 图片文件的 MIME 类型映射表
# 用于构造 data:image/xxx;base64 的 data URL 前缀
# 大模型需要根据 MIME 类型正确解码图片，必须与文件真实格式匹配
# ============================================================
_EXT_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


def _get_mime_type(filename: str) -> str:
    """
    根据文件后缀获取 MIME 类型，用于构造 data URL
    :param filename: 文件名（含后缀）
    :return: MIME 类型字符串，未知后缀默认 image/jpeg
    """
    ext = os.path.splitext(filename)[1].lower()
    return _EXT_TO_MIME.get(ext, "image/jpeg")


def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为 MinIO 支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回 True，否则 False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


# ============================================================
# 步骤 1：初始化 MD 核心数据
# 从 state 中获取 MD 文件路径 → 校验 → 读取内容 → 确定图片目录
# ============================================================

def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    从全局状态中提取并初始化 MD 处理所需核心数据
    上游节点 node_pdf_to_md 已将 md_path、md_content 写入 state
    :param state: 导入流程全局状态对象
    :return: 三元组 (MD文件内容, MD文件Path对象, 图片文件夹Path对象)
    :raise FileNotFoundError: 当状态中无有效 MD 文件路径时抛出
    """
    # 1. 获取 MD 文件路径，进行非空校验
    md_file_path = state['md_path']
    if not md_file_path:
        raise FileNotFoundError(f"全局状态中无有效MD文件路径：state['md_path']={repr(md_file_path)}")

    md_path_obj = Path(md_file_path)

    # 2. 校验文件是否存在
    if not md_path_obj.exists():
        raise FileNotFoundError(f"MD文件不存在：{md_path_obj.absolute()}")

    # 3. 获取 MD 内容：优先复用 state 中已有内容（上游节点已读取）
    #    若为空则从文件重新读取
    #    return 时 NameError。修复为统一从 state['md_content'] 取值
    if not state['md_content']:
        with md_path_obj.open('r', encoding='utf-8') as f:
            state['md_content'] = f.read()
        logger.debug(f"从文件读取MD内容完成，文件大小：{len(state['md_content'])} 字符")
    else:
        logger.debug(f"复用全局状态中的MD内容，内容大小：{len(state['md_content'])} 字符")

    # 4. 确定图片文件夹路径（MinerU 解析后图片固定在 MD 同级的 images/ 目录下）
    images_dir_obj = md_path_obj.parent / 'images'

    return state['md_content'], md_path_obj, images_dir_obj


# ============================================================
# 图片扫描辅助函数
# 在 MD 内容中正则匹配图片标签，提取上下文文本
# ============================================================

def find_image_in_md(md_content: str, image_file: str, context_len: int = 100) -> List[Tuple[str, str]]:
    """
    查找 MD 内容中指定图片的所有引用位置，并返回每个位置的上下文文本
    正则匹配 Markdown 图片语法：![描述](路径)
    使用非贪婪匹配 .*? 避免跨图片标签的「匹配过度」问题
    :param md_content: MD 文件完整内容
    :param image_file: 图片文件名（含后缀），如 "diagram.png"
    :param context_len: 上下文截取长度，默认前后各 100 字符
    :return: 上下文列表，每个元素为 (上文文本, 下文文本) 元组，无匹配则返回空列表
    """
    # 编译正则：匹配 markdown 图片标签 ![...](...图片文件名...)
    # re.escape(image_file) 转义文件名中的特殊字符（如 . + 等），防止正则语法错误
    # .*? 非贪婪匹配：尽可能少匹配字符，确保每个图片标签独立匹配，不会吃掉后面的标签
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
    results = []

    # finditer 迭代查找所有匹配项（适合大文本，逐个处理不占内存）
    for m in pattern.finditer(md_content):
        start, end = m.span()
        # 截取匹配位置的上文和下文，max/min 防止索引越界
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        logger.debug(f"图片 [{image_file}] 匹配到引用，上文：{pre_text.strip()}")
        logger.debug(f"图片 [{image_file}] 匹配到引用，下文：{post_text.strip()}")
        results.append((pre_text, post_text))

    if not results:
        logger.debug(f"MD内容中未找到图片 [{image_file}] 的引用")
    return results


# ============================================================
# 步骤 2：扫描图片文件夹
# 遍历 images/ 目录 → 过滤支持格式 → 校验 MD 引用 → 提取上下文
# ============================================================

def step_2_scan_images(md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    扫描图片文件夹，过滤出「支持格式 + MD中实际引用」的图片，组装处理元数据
    每张图片取第一个匹配到的上下文，后续步骤据此生成更精准的摘要
    :param md_content: MD 文件完整内容
    :param images_dir: 图片文件夹 Path 对象
    :return: 待处理图片列表，每个元素为三元组：
             (图片文件名, 图片完整路径字符串, (上文文本, 下文文本))
    """
    targets = []

    # 遍历图片文件夹中所有文件
    for image_file in os.listdir(images_dir):
        # 过滤 1：非支持格式的图片，跳过
        if not is_supported_image(image_file):
            logger.debug(f"图片格式不支持，跳过：{image_file}")
            continue

        # 过滤 2：查找该图片在 MD 中是否被实际引用，提取上下文
        context_list = find_image_in_md(md_content, image_file)
        if not context_list:
            logger.warning(f"图片未在MD中引用，跳过处理：{image_file}")
            continue

        # 组装待处理图片元数据
        # 取第一个匹配的上下文 (上文, 下文)，让 VLM 结合上下文生成更贴切的摘要
        img_path = str(images_dir / image_file)
        targets.append((image_file, img_path, context_list[0]))
        logger.info(f"图片加入待处理列表：{image_file}")

    logger.info(f"图片扫描完成，共筛选出待处理图片：{len(targets)} 张")
    return targets


# ============================================================
# 步骤 3：图片摘要生成
# 将图片编码为 Base64 → 构造多模态消息 → 调用 VLM 生成中文摘要
# 带 API 速率限制，防止并发过高触发风控
# ============================================================

def encode_image_to_base64(image_path: str) -> str:
    """
    将本地图片文件编码为 Base64 字符串
    VLM（视觉大模型）通过 data URL 方式接收图片，需 Base64 编码
    :param image_path: 图片本地完整路径
    :return: 图片的 Base64 编码字符串（UTF-8 解码）
    """
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    logger.debug(f"图片Base64编码完成，文件：{image_path}，编码后长度：{len(base64_str)}")
    return base64_str


def summarize_image(image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
    """
    调用多模态大模型生成图片内容摘要（适配 LangChain 工具类，复用项目统一 LLM 客户端）
    生成的摘要用于 Markdown 图片标题（Alt Text），使图片可被文本检索命中
    严格控制 50 字以内中文描述
    :param image_path: 图片本地完整路径
    :param root_folder: 文档所属文件夹 / 主名，为大模型提供上下文（如 "hl3040网络说明书"）
    :param image_content: 图片在 MD 中的上下文元组，格式 (上文文本, 下文文本)
    :return: 图片内容摘要（异常时返回默认值 "图片描述"）
    """
    # 将图片编码为 Base64，适配多模态大模型输入要求
    base64_image = encode_image_to_base64(image_path)

    # 根据文件后缀获取真实 MIME 类型（修复原代码硬编码 image/jpeg 的 bug）
    mime_type = _get_mime_type(image_path)

    try:
        # 1. 获取项目统一 LLM 客户端（自动缓存，传入多模态模型名如 qwen3-vl-flash）
        lvm_client = get_llm_client(model=lm_config.lv_model)

        # 2. 加载并渲染提示词（核心：传入所有占位符对应的变量）
        #    root_folder  → {root_folder}
        #    image_content → {image_content[0]}/{image_content[1]}
        prompt_text = load_prompt(
            name="image_summary",
            root_folder=root_folder,
            image_content=image_content
        )

        # 3. 构造 LangChain 标准多模态 HumanMessage（兼容千问 / OpenAI 等视觉模型）
        #    消息顺序：先图后文 或 先文后图 均可，视觉模型会自动对齐
        messages = [
            HumanMessage(
                content=[
                    # 文本提示词：携带上下文，限定摘要规则
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    # 多模态核心：Base64 编码图片数据（动态 MIME 类型）
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            )
        ]

        # 4. LangChain 标准调用：invoke 方法（工具类已封装超时/重试等参数）
        response = lvm_client.invoke(messages)

        # 5. 解析响应：去除首尾空白、合并换行符为空格（使摘要为单行，方便填入 alt 属性）
        #    response.content 在 LangChain 中类型为 str | list，此处确保以字符串方式处理
        raw_content = response.content
        if isinstance(raw_content, list):
            # 极少数情况下返回结构化列表，提取第一个文本内容
            raw_content = str(raw_content[0]) if raw_content else ""
        summary = str(raw_content).strip().replace("\n", " ")
        logger.info(f"图片摘要生成成功：{image_path}，摘要：{summary}")
        return summary

    except LangChainException as e:
        # LangChain 框架层异常（如模型名错误、API 格式不兼容）
        logger.error(f"图片摘要生成失败（LangChain框架异常）：{image_path}，错误信息：{str(e)}")
        return "图片描述"
    except Exception as e:
        # 其他系统级异常（网络超时、磁盘读取失败等）
        logger.error(f"图片摘要生成失败（系统异常）：{image_path}，错误信息：{str(e)}")
        return "图片描述"


def step_3_generate_summaries(
    doc_stem: str,
    targets: List[Tuple[str, str, Tuple[str, str]]],
    requests_per_minute: int = 9
) -> Dict[str, str]:
    """
    步骤 3：批量为待处理图片生成内容摘要，带 API 速率限制防止触发大模型限流
    设计思路：
    - 每张图片单独调用 VLM，异常时返回兜底值 "图片描述"，不阻塞后续图片处理
    - 滑动窗口限速（默认每分钟 9 次），避免并发过高被 API 风控拦截
    :param doc_stem: 文档文件名（不含后缀），作为大模型 prompt 上下文
    :param targets: 待处理图片列表，元素为 (图片文件名, 图片完整路径, 图片上下文)
    :param requests_per_minute: 每分钟最大 API 请求数，默认 9 次
    :return: 图片摘要字典，键：图片文件名，值：图片内容摘要字符串
    """
    summaries = {}
    # 初始化滑动窗口时间戳队列（跨循环复用，记录每次 API 调用的时间）
    request_times: deque = deque()

    for img_file, image_path, context in targets:
        # 速率限制：窗口内请求数超上限则自动阻塞等待
        # 这是「滑动窗口」算法，窗口时长 60 秒，最大 9 次请求
        apply_api_rate_limit(request_times, max_requests=requests_per_minute, window_seconds=60)

        logger.debug(f"开始生成图片摘要：{image_path}")
        # 调用 VLM 生成摘要（内部有 try/except 兜底，单张失败不影响整批）
        summaries[img_file] = summarize_image(
            image_path,
            root_folder=doc_stem,
            image_content=context
        )

    logger.info(f"图片摘要批量生成完成，共处理 {len(summaries)} 张图片")
    return summaries


# ============================================================
# 步骤 4：上传与替换
# 清理 MinIO 旧目录 → 批量上传图片 → 合并摘要和URL → 替换 MD 引用
# ============================================================

def clean_minio_directory(minio_client: Minio, prefix: str) -> None:
    """
    幂等性清理 MinIO 指定目录下的所有旧文件
    目的：
    - 同一个文档可能被重复导入，需清理旧图片防止内容混淆
    - 避免垃圾文件堆积，降低存储成本
    幂等性：多次调用结果一致，目录为空时不报错
    :param minio_client: 初始化完成的 MinIO 客户端对象
    :param prefix: MinIO 目录前缀（要清理的目录路径）
    """
    try:
        # 列出指定前缀下的所有对象（递归遍历子目录）
        objects_to_delete = minio_client.list_objects(
            bucket_name=minio_config.bucket_name,
            prefix=prefix,
            recursive=True
        )
        # 构造删除对象列表，过滤掉 object_name 为 None 的异常对象
        delete_list = [
            DeleteObject(obj.object_name)
            for obj in objects_to_delete
            if obj.object_name is not None
        ]

        if delete_list:
            logger.info(f"开始清理MinIO旧文件，待删除文件数：{len(delete_list)}，目录：{prefix}")
            # 批量删除对象（remove_objects 返回一个可迭代的错误列表）
            errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
            # 遍历删除错误信息，记录异常但不中断流程
            for error in errors:
                logger.error(f"MinIO文件删除失败：{error}")
        else:
            logger.debug(f"MinIO目录无旧文件，无需清理：{prefix}")
    except Exception as e:
        # 清理失败不中断流程（例如网络波动），记录错误后继续上传
        logger.error(f"MinIO目录清理失败：{prefix}，错误信息：{str(e)}")


def upload_to_minio(minio_client: Minio, local_path: str, object_name: str) -> str | None:
    """
    将单张本地图片上传至 MinIO 对象存储，并返回公网可访问 URL
    自动检测图片 MIME 类型，设置正确的 Content-Type 响应头
    :param minio_client: 初始化完成的 MinIO 客户端对象
    :param local_path: 图片本地完整路径
    :param object_name: MinIO 中要存储的对象名称（含目录路径）
    :return: 图片 MinIO 公网访问 URL（上传失败返回 None）
    """
    try:
        logger.info(f"开始上传图片至MinIO：本地路径={local_path}，MinIO对象名={object_name}")

        # 根据文件后缀推断 Content-Type（如 .png → image/png）
        # os.path.splitext 返回 (root, ext)，[1] 取后缀，[1:] 去掉开头的点
        content_type = f"image/{os.path.splitext(local_path)[1][1:]}"

        # fput_object：文件流上传，适合大文件，自动分块传输
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=object_name,
            file_path=local_path,
            content_type=content_type
        )

        # 处理路径中的反斜杠（Windows 路径），替换为 URL 编码 %5C
        object_name_escaped = object_name.replace("\\", "%5C")
        # 根据配置选择 HTTP/HTTPS 协议
        protocol = "https" if minio_config.minio_secure else "http"
        # 构造 MinIO 公网可访问 URL
        # 格式：http(s)://{endpoint}/{bucket_name}/{object_name}
        base_url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}"

        # 确保 object_name 不以 / 开头（MinIO 对象名规范）
        # 同时中间不需要双斜杠
        clean_object = object_name_escaped.lstrip("/")
        img_url = f"{base_url}/{clean_object}"

        logger.info(f"图片上传成功，访问URL：{img_url}")
        return img_url
    except Exception as e:
        logger.error(f"图片上传MinIO失败：{local_path}，错误信息：{str(e)}")
        return None


def upload_images_batch(
    minio_client: Minio,
    upload_dir: str,
    targets: List[Tuple[str, str, Tuple[str, str]]]
) -> Dict[str, str]:
    """
    批量上传待处理图片至 MinIO，返回图片文件名与访问 URL 的映射关系
    每张图片独立上传，单张失败不影响其他图片
    :param minio_client: 初始化完成的 MinIO 客户端对象
    :param upload_dir: MinIO 上传根目录（如 "/upload-images/hl3040网络说明书"）
    :param targets: 待处理图片列表，元素为 (图片文件名, 图片完整路径, 图片上下文)
    :return: 图片 URL 字典，键：图片文件名，值：MinIO 访问 URL
    """
    urls = {}
    for img_file, img_path, _ in targets:
        # 构造 MinIO 对象名称：目录 + 文件名
        object_name = f"{upload_dir}/{img_file}"
        logger.debug(f"构造MinIO对象名称：{object_name}")

        # 海象运算符 := ：在表达式内完成「赋值 + 判断」
        # 如果上传成功返回 URL，存入 urls；失败返回 None，自动跳过
        if img_url := upload_to_minio(minio_client, img_path, object_name):
            urls[img_file] = img_url

    logger.info(f"图片批量上传完成，成功上传 {len(urls)}/{len(targets)} 张图片")
    return urls


def merge_summary_and_url(summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """
    合并图片摘要字典和 URL 字典，过滤掉上传失败（无 URL）的图片
    这一步是连接「摘要生成」和「MD 内容替换」的桥梁
    :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
    :param urls: 图片 URL 字典，键：图片文件名，值：MinIO 访问 URL
    :return: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL) 元组
    """
    image_info = {}
    # 以 summaries 为主导遍历，因为可能有图片摘要生成成功但上传失败
    # 只有同时拥有摘要和 URL 的图片才参与后续 MD 替换
    for image_file, summary in summaries.items():
        if url := urls.get(image_file):  # 海象运算符：赋值 + 判真
            image_info[image_file] = (summary, url)

    logger.info(f"图片摘要与URL合并完成，有效图片信息 {len(image_info)} 条")
    return image_info


def process_md_file(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
    """
    核心功能：替换 MD 内容中的本地图片引用为 MinIO 远程引用
    替换规则：![原描述](本地路径) → ![图片摘要](MinIO访问URL)
    使用 Lambda 替换避免 summary 或 URL 中的特殊字符（如反斜杠、$等）
    被正则引擎误解为正则语法
    :param md_content: 原始 MD 文件内容
    :param image_info: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)
    :return: 替换后的新 MD 内容
    """
    for img_filename, (summary, new_url) in image_info.items():
        # 正则匹配 MD 图片标签，忽略大小写，兼容不同路径写法
        # 规则：![任意描述](任意路径+图片文件名+任意后缀)
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_filename) + r".*?\)",
            re.IGNORECASE
        )

        # 使用 Lambda 进行替换（防御性编程）：
        # - 如果 summary / new_url 包含反斜杠 \、美元符号 $ 等正则特殊字符，
        #   直接字符串替换 re.sub("str", ...) 会出错，Lambda 则安全绕过
        # - Lambda 每次匹配时都会被调用，生成替换字符串
        md_content = pattern.sub(
            lambda _, s=summary, u=new_url: f"![{s}]({u})",
            md_content
        )
        logger.debug(f"完成MD图片引用替换：{img_filename} → {new_url}")

    logger.info(f"MD文件图片引用替换完成，共替换 {len(image_info)} 处图片引用")
    return md_content


def step_4_upload_and_replace(
    minio_client: Minio,
    doc_stem: str,
    targets: List[Tuple[str, str, Tuple[str, str]]],
    summaries: Dict[str, str],
    md_content: str
) -> str:
    """
    步骤 4：核心流程 - 图片上传 MinIO + 合并摘要 &URL + 替换 MD 图片引用
    完整流程：
    1. 清理 MinIO 旧目录（幂等性，解决重复导入问题）
    2. 批量上传新图片（每张独立，失败不阻塞）
    3. 合并摘要和 URL（过滤上传失败的图片）
    4. 替换 MD 内容中的本地图片引用为远程 URL
    :param minio_client: 初始化完成的 MinIO 客户端对象
    :param doc_stem: 文档文件名（不含后缀），作为 MinIO 子目录名（按文档隔离）
    :param targets: 待处理图片列表
    :param summaries: 图片摘要字典
    :param md_content: 原始 MD 文件内容
    :return: 图片引用替换后的新 MD 内容
    """
    # 构造 MinIO 上传目录：配置根目录 + 文档主名（去空格，避免路径问题）
    minio_img_dir = minio_config.minio_img_dir
    upload_dir = f"{minio_img_dir}/{doc_stem}".replace(" ", "")

    # 步骤 4.1：清理该文档对应的 MinIO 旧目录（幂等性保证）
    clean_minio_directory(minio_client, upload_dir)

    # 步骤 4.2：批量上传图片至 MinIO，获取文件名→URL 映射
    urls = upload_images_batch(minio_client, upload_dir, targets)

    # 步骤 4.3：合并图片摘要和 URL，过滤上传失败的图片
    image_info = merge_summary_and_url(summaries, urls)

    # 步骤 4.4：替换 MD 内容中的本地图片引用为 MinIO 远程引用
    if image_info:
        md_content = process_md_file(md_content, image_info)
    else:
        logger.warning("无有效图片信息，跳过MD内容替换")

    return md_content


# ============================================================
# 步骤 5：备份与保存
# 将处理后内容保存为 _new.md，原文件保持不变，更新 state 中的路径
# ============================================================

def step_5_backup_new_md_file(origin_md_path: str, md_content: str) -> str:
    """
    步骤 5：将处理后的 MD 内容保存为新文件（原文件不变，避免数据丢失）
    新文件命名规则：原文件名 + _new.md（如 test.md → test_new.md）
    这是整个节点链路的「落盘」步骤，确保处理结果持久化
    :param origin_md_path: 原始 MD 文件完整路径
    :param md_content: 处理后的新 MD 内容（图片路径已替换为 MinIO URL + 摘要）
    :return: 新 MD 文件的完整路径
    """
    # 构造新文件路径：去除原后缀 → 拼接 _new.md
    # 例如：/output/hl3040/hl3040网络说明书.md → /output/hl3040/hl3040网络说明书_new.md
    new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"

    # 写入新 MD 内容（覆盖写入，若文件已存在则更新）
    with open(new_md_file_name, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"处理后MD文件已保存，新文件路径：{new_md_file_name}")
    return new_md_file_name


# ============================================================
# 节点主入口：node_md_img
# LangGraph 节点函数，编排 5 个步骤完成图片全流程处理
# ============================================================

def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    MD 文件图片处理核心节点 — 五步法完成图片全流程处理
    节��定位：
    - 上游：node_pdf_to_md（已完成 PDF→Markdown 转换，state 中有 md_path、md_content）
    - 下游：node_document_split（需要处理后的新 MD 文件进行文档切分）
    核心流程：
    1. 初始化获取 MD 内容、文件路径、图片文件夹路径
    2. 扫描图片文件夹，筛选 MD 中实际引用的支持格式图片
    3. 调用多模态大模型为图片生成内容摘要（50字内中文描述）
    4. 将图片上传至 MinIO，替换 MD 中本地图片路径为 MinIO 访问 URL，并填充图片摘要
    5. 备份原 MD 文件，保存处理后的新 MD 文件并更新 state 中的路径
    :param state: 导入流程全局状态对象，包含 task_id、md_path、md_content 等核心参数
    :return: 更新后的全局状态对象（md_path → _new.md，md_content → 替换后的新内容）
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行，当前状态关键字段：task_id={state.get('task_id')}")
    add_running_task(state['task_id'], function_name)

    try:
        # ========================================
        # 步骤 1：校验并获取本次操作的数据
        # 从 state 提取 md_path → 读取 md_content → 确定 images/ 目录
        # ========================================
        md_content, md_path_obj, images_dir_obj = step_1_get_content(state)

        # 无图片文件夹，跳过所有图片处理（非错误，正常分支）
        if not images_dir_obj.exists():
            logger.info(f">>> [{function_name}] 图片文件夹不存在，跳过图片处理：{images_dir_obj.absolute()}")
            return state

        # 初始化 MinIO 客户端（模块级单例，首次连接，后续复用）
        minio_client = get_minio_client()
        if not minio_client:
            # MinIO 连接失败：无法上传图片，但不应终止整个导入流程
            # 下游节点仍可基于原始 MD 继续处理（只是图片无远程 URL）
            logger.warning(f">>> [{function_name}] MinIO客户端初始化失败，已跳过图片处理全流程")
            return state

        # ========================================
        # 步骤 2：识别 MD 中使用过的图片
        # 扫描 images/ → 过滤未引用/非支持格式 → 提取上下文
        # ========================================
        targets = step_2_scan_images(md_content, images_dir_obj)
        if not targets:
            logger.info(f">>> [{function_name}] 未检测到MD中引用的支持格式图片，跳过后续处理")
            return state

        # ========================================
        # 步骤 3：进行图片内容的总结和处理（视觉模型）
        # 每张图片：Base64 编码 → 多模态消息 → VLM 生成摘要
        # 带滑动窗口 API 速率限制（默认 9 次/分钟）
        # ========================================
        summaries = step_3_generate_summaries(md_path_obj.stem, targets)

        # ========================================
        # 步骤 4：上传图片 minio 以及更新 md 的内容
        # 清旧目录 → 批量上传 → 合并 URL+摘要 → 替换 MD 引用
        # 替换后：![原描述](本地路径) → ![图片摘要](MinIO远程URL)
        # ========================================
        new_md_content = step_4_upload_and_replace(
            minio_client, md_path_obj.stem, targets, summaries, md_content
        )
        state['md_content'] = new_md_content

        # ========================================
        # 步骤 5：进行数据的最终处理和备份
        # 原始 MD 不变，内容写入 _new.md 文件，更新 state 路径
        # 下游节点 node_document_split 将读取新文件进行切分
        # ========================================
        new_md_path = step_5_backup_new_md_file(state['md_path'], new_md_content)
        state['md_path'] = new_md_path

        logger.info(f">>> [{function_name}] 节点执行完成，新MD文件：{new_md_path}")
        return state

    finally:
        # 无论正常完成还是异常退出，都要向前端推送节点结束状态
        # finally 确保 add_done_task 一定被调用（包括早期 return 路径）
        add_done_task(state['task_id'], function_name)


# ============================================================
# 单元测试入口
# 直接运行 python -m app.import_process.agent.nodes.node_md_img
# ============================================================

if __name__ == "__main__":
    """
    本地测试入口：单独运行该文件时，执行 MD 图片处理全流程测试
    测试前提：
    - 项目根目录的 output/ 下存在处理后的 MD 文件及其 images/ 子目录
    - .env 中已配置 MinIO 和 LLM 的连接信息
    """
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试 MD 文件路径（需存在：output/hak180产品安全手册/hak180产品安全手册.md）
    # 小文件测试（6张图），hl3040有207张图，跑全流程约3~4分钟
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟上游节点传入的 state
        # md_content 设为空字符串，step_1 会自动从文件读取
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：md_path={result_state.get('md_path')}")
        logger.info(f"本地测试完成 - md_content 前500字符：{result_state.get('md_content', '')[:500]}")
