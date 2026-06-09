# 加载环境变量：从 .env 文件读取配置（如Milvus地址、KG服务地址、BGE模型路径等）
from dotenv import load_dotenv
# 导入LangGraph核心依赖：StateGraph(状态图)、START/END(内置起始/结束节点常量)
from langgraph.graph import StateGraph, END, START

from app.core.logger import logger
# 导入自定义状态类：统一管理工作流全程的所有数据（各节点共享/修改）
from app.import_process.agent.state import ImportGraphState, create_default_state
# 导入所有自定义业务节点：每个节点对应知识库导入的一个具体步骤
from app.import_process.agent.nodes.node_entry import node_entry  # 入口节点：初始化参数、校验输入
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md  # PDF转MD：解析PDF文件为markdown格式
from app.import_process.agent.nodes.node_md_img import node_md_img  # MD图片处理：提取/下载markdown中的图片、修复图片路径
from app.import_process.agent.nodes.node_document_split import node_document_split  # 文档分块：将长文档切分为符合模型要求的小片段
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition  # 项目名识别：从分块中提取核心项目名称（业务定制化）
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  # BGE向量化：将文本分块转换为向量表示（适配Milvus向量库）
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus  # 导入Milvus：将向量数据写入Milvus向量数据库


# 初始化环境变量：必须在配置读取前执行，确保后续节点能获取到环境变量中的配置信息
load_dotenv()

# 1. 初始化Langgraph状态图
workflow = StateGraph(ImportGraphState)
# 2. 注册所有的子节点
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)
# 3. 设置入口节点
workflow.add_edge(START, "node_entry")
# 4. 定义条件边的路由函数
def route_after_entry(state:ImportGraphState) -> str:
    """
    根据文件类型判断第二个节点的路线
        文件是pdf -> node_pdf_to_md
        文件是md ->  node_md_img
        既不是文件 又不是md -> END
    :param state: is_md_read_enabled is_pdf_read_enadled
    :return: node_pdf_to_md | node_md_img | END
    """
    if state["is_md_read_enabled"]:
        return "node_pdf_to_md"
    elif state["is_pdf_read_enabled"]:
        return "node_md_img"
    else:
        return END
workflow.add_conditional_edges(
"node_entry",
        route_after_entry,{
        "node_pdf_to_md": "node_pdf_to_md",
        "node_md_img": "node_md_img",
        END : END,
    })
# 5. 定义静态边
workflow.add_edge("node_pdf_to_md", "node_md_img")  # PDF转MD完成 → MD图片处理
workflow.add_edge("node_md_img", "node_document_split")  # MD处理完成 → 文档分块
workflow.add_edge("node_document_split", "node_item_name_recognition")  # 分块完成 → 项目名识别
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")  # 项目名识别完成 → BGE向量化
workflow.add_edge("node_bge_embedding", "node_import_milvus")  # 向量化完成 → 导入Milvus向量库
workflow.add_edge("node_import_milvus", END)  # Milvus入库完成 → 工作流执行结束（END是内置结束节点）
# 6. 编译图节点对象
kb_import_app = workflow.compile()