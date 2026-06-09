# 🧠 掌柜智库 — 企业级多模态 RAG 智能问答系统

基于 LangGraph + Milvus + BGE-M3 构建的私有知识库问答系统，支持 PDF 说明书自动解析入库、多路混合检索、Cross-Encoder 精排、LLM 流式生成，解决大模型在垂直领域的"幻觉"与知识滞后问题。

---

## 🏗️ 系统架构

```
┌─────────────────────────── 导入链路 (Ingestion) ───────────────────────────┐
│                                                                             │
│  PDF → MinerU 解析 → Markdown → VLM 图片摘要 → 智能切分 → LLM 商品名识别  │
│                                                    ↓                       │
│                                          BGE-M3 双向量化 → Milvus 入库      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── 查询链路 (Retrieval) ───────────────────────────┐
│                                                                             │
│  用户问题 → Query 改写 → 四路并行召回 ─┐                                    │
│        (LLM 消解指代)    ├─ 稠密+稀疏混合检索                               │
│                          ├─ HyDE 假设文档检索                               │
│                          ├─ MCP 联网搜索                                    │
│                          └─ 知识图谱查询                                    │
│                                    ↓                                        │
│                          RRF 倒排秩融合 → BGE Reranker 精排 → LLM 流式生成  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 核心技术栈

| 类别 | 技术 |
|---|---|
| **图编排引擎** | LangGraph（有向状态图，双链路并行调度） |
| **大模型** | Qwen-Flash / Qwen3-VL-Flash（阿里百炼 API） |
| **向量模型** | BGE-M3（稠密 1024 维 + 稀疏变长向量，单次推理双输出） |
| **向量数据库** | Milvus（HNSW 索引 + WeightedRanker 混合检索） |
| **精排模型** | BGE Reranker Large（Cross-Encoder 语义精排） |
| **PDF 解析** | MinerU（云端 API，复杂排版还原为 Markdown） |
| **图片理解** | Qwen3-VL-Flash（VLM 看图说话，图片 → 可搜索文字） |
| **对象存储** | MinIO（图片持久化） |
| **对话历史** | MongoDB（多轮会话上下文） |
| **联网搜索** | 百炼 MCP WebSearch（Streamable HTTP 协议） |
| **Web 服务** | FastAPI + SSE + BackgroundTasks（异步 + 流式） |

---

## 📁 项目结构

```
RAG_py/
├── app/
│   ├── import_process/          # 导入链路
│   │   ├── agent/
│   │   │   ├── main_graph.py    # 7 节点 DAG（入口 → PDF解析 → 图片 → 切分 → 识别 → 向量化 → 入库）
│   │   │   ├── state.py         # ImportGraphState 状态定义
│   │   │   ├── test_graph_flow.py
│   │   │   └── nodes/
│   │   │       ├── node_entry.py               # ① 入口校验 + 路由分发
│   │   │       ├── node_pdf_to_md.py           # ② MinerU PDF → Markdown
│   │   │       ├── node_md_img.py              # ③ VLM 图片摘要 + MinIO 上传
│   │   │       ├── node_document_split.py      # ④ 按标题层级智能切分
│   │   │       ├── node_item_name_recognition.py # ⑤ LLM 商品名识别
│   │   │       ├── node_bge_embedding.py       # ⑥ BGE-M3 稠密+稀疏双向量化
│   │   │       └── node_import_milvus.py       # ⑦ Milvus 自动建表 + 幂等入库
│   │   ├── api/
│   │   │   └── file_import_service.py          # FastAPI 文件上传服务
│   │   └── page/
│   │       └── import.html                     # 文件上传前端页面
│   │
│   ├── query_process/           # 查询链路
│   │   ├── agent/
│   │   │   ├── main_graph.py    # 7 节点 DAG（改写 → 四路并行 → RRF → Rerank → 生成）
│   │   │   ├── state.py         # QueryGraphState 状态定义
│   │   │   └── nodes/
│   │   │       ├── node_item_name_confirm.py   # ① Query 改写 + 商品名确认
│   │   │       ├── node_search_embedding.py    # ② BGE-M3 混合向量检索
│   │   │       ├── node_search_embedding_hyde.py # ③ HyDE 假设文档检索
│   │   │       ├── node_web_search_mcp.py      # ④ MCP 联网搜索
│   │   │       ├── node_query_kg.py            # ⑤ 知识图谱查询
│   │   │       ├── node_rrf.py                 # ⑥ RRF 多路融合排序
│   │   │       ├── node_rerank.py              # ⑦ BGE Reranker Cross-Encoder 精排
│   │   │       └── node_answer_output.py       # ⑧ LLM 流式生成答案
│   │   ├── api/
│   │   │   └── query_server.py                 # FastAPI 问答服务（支持 SSE 流式）
│   │   └── page/
│   │       └── chat.html                       # 对话前端页面
│   │
│   ├── clients/                 # 外部服务客户端
│   │   ├── milvus_utils.py      # Milvus 连接 + 混合搜索
│   │   ├── minio_utils.py       # MinIO 客户端
│   │   └── mongo_history_utils.py # MongoDB 对话历史读写
│   │
│   ├── lm/                      # 模型封装
│   │   ├── lm_utils.py          # LLM 客户端封装
│   │   ├── embedding_utils.py   # BGE-M3 向量化封装
│   │   └── reranker_utils.py    # BGE Reranker 封装
│   │
│   ├── conf/                    # 配置
│   │   ├── milvus_config.py
│   │   ├── reranker_config.py
│   │   └── bailian_mcp_config.py
│   │
│   ├── core/                    # 核心模块
│   │   ├── logger.py            # 日志（基于 loguru）
│   │   └── load_prompt.py       # 提示词加载
│   │
│   └── utils/                   # 工具模块
│       ├── task_utils.py        # 任务状态管理
│       ├── sse_utils.py         # SSE 推送工具
│       ├── path_util.py         # 路径工具
│       └── escape_milvus_string_utils.py
│
├── prompts/                     # 提示词模板
├── doc/                         # 测试文档
├── output/                      # 处理结果输出
├── logs/                        # 日志文件
├── .env                         # 环境变量（不提交 Git）
├── .env.example                 # 环境变量模板
├── pyproject.toml               # 依赖管理
├── OVERVIEW.md                  # 项目全景文档
├── STARTUP.md                   # 启动指南
└── README.md                    # 本文件
```

---

## ⚡ 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/xsummer624-gif/xiasunny.git
cd xiasunny
```

### 2. 安装依赖

```bash
uv sync
```

### 3. 环境配置

复制 `.env.example` 为 `.env`，填写以下必要配置：

```bash
# 必填：百炼 API Key
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 必填：BGE-M3 向量模型路径
BGE_M3_PATH=/path/to/models/BAAI/bge-m3

# 必填：BGE Reranker 模型路径
BGE_RERANKER_LARGE=/path/to/models/BAAI/bge-reranker-large

# 必填：MinerU API Token
MINERU_API_TOKEN=xxx
```

### 4. 启动基础设施

按顺序启动依赖服务：

```bash
# ① 启动 etcd（Milvus 元数据存储）
~/milvus/etcd-v3.5.18-windows-amd64/etcd.exe

# ② 启动 Milvus（向量数据库）
cd ~/milvus && docker compose up -d

# ③ 启动 MinIO（对象存储）
docker start minio

# ④ 启动 MongoDB（对话历史）
net start mongodb
```

详细步骤见 [STARTUP.md](STARTUP.md)。

### 5. 启动应用

```bash
# 导入服务（文件上传页面）
uv run python -m app.import_process.api.file_import_service

# 查询服务（AI 对话页面）
uv run python -m app.query_process.api.query_server
```

浏览器访问：
- **文件上传**：`http://127.0.0.1:8000/import.html`
- **AI 对话**：`http://127.0.0.1:8001/chat.html`

---

## 🔄 数据流详解

### 导入链路（7 节点串行）

```
START
  │
  ▼
① node_entry          — 校验 PDF 路径，提取文件名，设置路由标志
  │
  ▼
② node_pdf_to_md      — 调用 MinerU API 将 PDF 转为 Markdown（含图片引用）
  │
  ▼
③ node_md_img         — Qwen3-VL 为每张图片生成 50 字中文摘要，上传 MinIO
  │
  ▼
④ node_document_split — 按 Markdown 标题层级切分，2000 字/块，<500 字合并
  │
  ▼
⑤ node_item_name_recognition — LLM 从前 5 个 Chunk 提取商品名，写入 kb_item_names
  │
  ▼
⑥ node_bge_embedding  — BGE-M3 分批生成 1024 维稠密向量 + 变长稀疏向量
  │
  ▼
⑦ node_import_milvus  — 自动建表（HNSW 索引），幂等删除旧数据，批量插入
  │
  ▼
 END
```

### 查询链路（8 节点 + 条件分支 + 并行召回）

```
START
  │
  ▼
① node_item_name_confirm    — LLM 改写 Query + MongoDB 读历史 + Milvus 确认商品名
  │                     ┌── answer 非空 → 反问用户 / 拒绝 → 跳到 ⑧
  │ 条件路由 ──────────┤
  │                     └── answer 为空 → 继续检索
  │
  ▼ 四路并行召回
② node_search_embedding      — BGE-M3 混合搜索 kb_chunks（稠密 COSINE + 稀疏 IP）
③ node_search_embedding_hyde — LLM 生成假设文档 → 向量化 → 检索（提升语义稀疏场景）
④ node_web_search_mcp        — 百炼 MCP 联网搜索（外部实时信息）
⑤ node_query_kg              — Neo4j 知识图谱查询
  │
  ▼ 汇合
⑥ node_rrf                   — RRF 倒排秩融合（四条路排名加权，k=60）
  │
  ▼
⑦ node_rerank                — BGE Reranker Cross-Encoder 精排 Top 5
  │
  ▼
⑧ node_answer_output         — Prompt 拼装 → LLM 流式生成 → SSE 推送 → MongoDB 存档
  │
  ▼
 END
```

---

## ✨ 核心亮点

### 1. 双向量检索
BGE-M3 单次推理同时产出 **1024 维稠密向量**（COSINE）和**变长稀疏向量**（IP），Milvus WeightedRanker 以 0.8/0.2 权重融合——语义匹配 + 关键词匹配互补。

### 2. 三级检索漏斗
```
多路召回 (Top 50) → RRF 融合 (Top 10) → Cross-Encoder 精排 (Top 5)
```
每一级做不同的事：召回保覆盖、融合去分歧、精排定质量。

### 3. 幂等设计
同一个商品重复导入时，先按 `item_name` 删旧数据再插入新数据，避免搜索结果出现新旧混合。

### 4. 图片可搜索化
PDF 中的图片经 Qwen3-VL 生成中文摘要后嵌入 Chunk，BGE-M3 向量化——用户搜"怎么接线"能命中接线图。

### 5. 容错降级
- LLM 超时 → file_title 兜底，不阻塞整条链路
- MinIO 上传失败 → 仅在日志告警，本地文件正常处理
- Chunk 异常批次 → 保留原始数据，不影响其他批次

---

## 🧪 测试

```bash
# 单节点测试
uv run python -m app.import_process.agent.nodes.node_pdf_to_md
uv run python -m app.import_process.agent.nodes.node_import_milvus

# 全流程 DAG 测试
uv run python -m app.query_process.agent.test_graph_flow

# API 接口测试
curl http://127.0.0.1:8001/health
curl -X POST http://127.0.0.1:8001/query -H "Content-Type: application/json" -d '{"query":"HAK 180 怎么用","is_stream":false}'
```

---

## 🔧 技术选型说明

| 选择 | 原因 |
|---|---|
| BGE-M3 vs text-embedding-3 | 单次推理双向量输出，1024 维比 1536 维存储更小，支持中英双语 |
| LangGraph vs Chain | 有向图支持条件分支 + 并行 fan-out，Chain 只能串行 |
| MinerU vs PyPDF2 | 复杂排版（双栏、表格、公式）PyPDF2 丢失率超 30%，MinerU 基于大模型准确还原 |
| Milvus vs FAISS | 分布式部署、双向量索引、自动扩缩容，FAISS 是单机库 |
| HNSW vs IVF | 查询延迟 2ms vs 10ms，适合低延迟 RAG 场景 |

---

## 📄 许可证

MIT License
