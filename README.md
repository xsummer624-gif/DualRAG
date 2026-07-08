# 掌柜智库 — 企业级双模 RAG 智能问答系统

基于 LangGraph + Milvus + BGE-M3 构建的私有知识库问答系统，支持 **传统固定管线** 与 **Agentic 自主决策** 两种查询模式，共享同一套知识库与基础设施。

---

## 系统架构

```
┌─────────────────── 导入链路 (Ingestion, :8000) ───────────────────┐
│                                                                     │
│  PDF → MinerU 解析 → Markdown → VLM 图片摘要 → 智能切分 → LLM 商品名│
│                                                    ↓               │
│                                          BGE-M3 双向量化 → Milvus 入库│
└─────────────────────────────────────────────────────────────────────┘

┌─────────── 查询链路 — 传统模式 (Retrieval, :8001) ───────────────┐
│                                                                     │
│  用户问题 → Query 改写 → 四路并行召回 ─┐                            │
│        (LLM 消解指代)    ├─ 稠密+稀疏混合检索                       │
│                          ├─ HyDE 假设文档检索                       │
│                          ├─ MCP 联网搜索                            │
│                          └─ 知识图谱查询                            │
│                                    ↓                                │
│                          RRF → BGE Reranker 精排 → LLM 流式生成     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────── 查询链路 — Agentic 模式 (Agentic, :8002) ─────────────┐
│                                                                     │
│  用户问题 → [强制主题识别] → Agent 自主循环 (≤3次工具调用) → 综合回答│
│                  │                                                   │
│      ┌───────────┼───────────┐                                      │
│      ▼           ▼           ▼                                      │
│  KB 搜索    HyDE 搜索   MCP 联网搜索                                 │
│  (复用传统)  (复用传统)  (复用传统)                                  │
│                                                                     │
│  工具按需调用 → LLM 自主判断 → 引用来源输出                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心技术栈

| 类别 | 技术 |
|---|---|
| **图编排引擎** | LangGraph（传统 DAG + Agentic create_react_agent） |
| **大模型** | Qwen-Flash / Qwen3-VL-Flash（阿里百炼 API） |
| **向量模型** | BGE-M3（稠密 1024 维 + 稀疏变长向量，单次推理双输出） |
| **向量数据库** | Milvus（HNSW 索引 + WeightedRanker 混合检索） |
| **精排模型** | BGE Reranker Large（Cross-Encoder 语义精排，仅传统模式） |
| **PDF 解析** | MinerU（云端 API，复杂排版还原为 Markdown） |
| **图片理解** | Qwen3-VL-Flash（VLM 看图说话，图片 → 可搜索文字） |
| **对象存储** | MinIO（图片持久化） |
| **对话历史** | MongoDB（多轮会话上下文） |
| **联网搜索** | 百炼 MCP WebSearch（Streamable HTTP 协议） |
| **Web 服务** | FastAPI + SSE + BackgroundTasks（异步 + 流式） |

---

## 项目结构

```
RAG_py/
├── app/
│   ├── agentic_query/          # Agentic 查询链路（新增）
│   │   ├── agent/
│   │   │   ├── agentic_graph.py    # create_react_agent + 主题识别预处理
│   │   │   └── tools/
│   │   │       ├── kb_search.py      # 复用传统 RAG 的向量搜索
│   │   │       ├── hyde_search.py    # 复用传统 RAG 的 HyDE 搜索
│   │   │       └── web_search.py     # 复用传统 RAG 的 MCP 搜索
│   │   ├── api/
│   │   │   └── agentic_server.py  # FastAPI 服务（端口 8002）
│   │   └── page/
│   │       └── chat.html          # Agentic 对话前端（含调用链可视化）
│   │
│   ├── query_process/           # 传统查询链路
│   │   ├── agent/
│   │   │   ├── main_graph.py    # 8 节点 DAG
│   │   │   └── nodes/           # 各节点函数，部分被 Agentic 复用
│   │   └── api/
│   │       └── query_server.py  # FastAPI 问答服务（端口 8001）
│   │
│   ├── import_process/          # 导入链路
│   │   ├── agent/
│   │   │   ├── main_graph.py    # 7 节点 DAG
│   │   │   └── nodes/
│   │   └── api/
│   │       └── file_import_service.py  # FastAPI 上传服务（端口 8000）
│   │
│   ├── evaluation/              # RAG 评估
│   │   └── ragas_metrics.py     # 自实现 Faithfulness / Answer Relevancy / Context Precision
│   │
│   ├── clients/                 # 外部服务客户端
│   ├── lm/                      # 模型封装
│   ├── conf/                    # 配置
│   ├── core/                    # 核心模块
│   └── utils/                   # 工具模块
│
├── prompts/                     # 提示词模板
├── logs/                        # 日志文件
├── .env                         # 环境变量
├── pyproject.toml               # 依赖管理
└── README.md                    # 本文件
```

---

## 三种服务

| 端口 | 服务 | 入口 | 说明 |
|---|---|---|---|
| 8000 | 导入服务 | `app/import_process/api/file_import_service.py` | PDF 上传、解析、切分、向量化、入库 |
| 8001 | 传统 RAG | `app/query_process/api/query_server.py` | 固定 8 节点 DAG：改写 → 并行召回 → 融合 → 精排 → 生成 |
| 8002 | Agentic RAG | `app/agentic_query/api/agentic_server.py` | 主题识别预处理 + Agent 自主决策循环 |

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/xsummer624-gif/rag.git
cd rag
```

### 2. 安装依赖

```bash
uv sync
```

或激活已有虚拟环境：

```bash
.venv\Scripts\activate
```

### 3. 环境配置

复制 `.env.example` 为 `.env`，填写必要配置。

### 4. 启动基础设施

按顺序启动依赖服务（详见 [STARTUP.md](STARTUP.md)）：
- etcd（Milvus 元数据存储）
- Milvus（向量数据库）
- MinIO（对象存储）
- MongoDB（对话历史）

### 5. 启动应用

```bash
# 导入服务（端口 8000）
uv run python -m app.import_process.api.file_import_service

# 传统 RAG 查询（端口 8001）
uv run python -m app.query_process.api.query_server

# Agentic RAG 查询（端口 8002）
uv run python -m app.agentic_query.api.agentic_server
```

浏览器访问：
- **文件上传**：`http://127.0.0.1:8000/`
- **传统 RAG**：`http://127.0.0.1:8001/chat.html`
- **Agentic RAG**：`http://127.0.0.1:8002/chat.html`

---

## 数据流详解

### 导入链路（7 节点串行）

```
① node_entry → ② node_pdf_to_md → ③ node_md_img
  → ④ node_document_split → ⑤ node_item_name_recognition
  → ⑥ node_bge_embedding → ⑦ node_import_milvus
```

PDF 文件 → MinerU 转 Markdown → VLM 图片摘要 → 按标题切分（2000 字/块） → LLM 提取商品名 → BGE-M3 双向量化 → MinIO + Milvus 入库。

### 传统 RAG 查询链路（8 节点 DAG）

```
① node_item_name_confirm ─── 条件路由 ──→ ⑧ node_answer_output
        │ (确认商品名)              (反问/拒绝时跳转)
        ▼ 四路并行
② node_search_embedding       ③ node_search_embedding_hyde
④ node_web_search_mcp         ⑤ node_query_kg
        ▼
⑥ node_rrf (RRF 融合) → ⑦ node_rerank (BGE 精排) → ⑧ node_answer_output (LLM 生成)
```

### Agentic RAG 查询链路

```
┌─────────────────────────────────────────────────────┐
│ ① 强制主题识别（固定执行，不经过 Agent 判断）       │
│    step_3_extract_info → LLM 提取商品名+改写问题     │
│    step_4_vectorize_and_query → Milvus 向量对齐      │
│    step_5_align_item_names → 确认标准化商品名        │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ ② Agent 自主决策循环（create_react_agent）           │
│                                                     │
│   System Prompt（含主题识别结果动态注入）              │
│          │                                           │
│   LLM 判断 → 调用工具 or 直接回答                    │
│          │                                           │
│   ┌──────┴──────┐                                    │
│   ▼              ▼                                   │
│ 调用工具       生成最终回答                           │
│   │              │                                   │
│   ▼              │                                   │
│ 工具返回        Done                                 │
│ (截断 300 字)   │                                   │
│   │              │                                   │
│   └──────┬──────┘                                    │
│          ▼                                           │
│   LLM 重新判断：够了吗？                             │
│   → 不够 → 继续调工具（最多 3 次）                   │
│   → 够了 → 生成最终回答                              │
│                                                     │
│   工具列表：                                         │
│   • search_knowledge_base       (复用传统节点)       │
│   • search_knowledge_base_enhanced (复用传统节点)    │
│   • search_web                  (复用传统节点)       │
└─────────────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│ ③ 后处理                                             │
│    RAGAS 评估（Faithfulness / Answer Relevancy /    │
│    Context Precision）                               │
│    调用链可视化（前端 chip 展示 + hover 查看详情）   │
│    保存对话历史到 MongoDB                            │
└─────────────────────────────────────────────────────┘
```

---

## 传统 RAG vs Agentic RAG

| 维度 | 传统 RAG（:8001） | Agentic RAG（:8002） |
|---|---|---|
| **管线结构** | 固定 8 节点 DAG，串行 + 并行 | Agent 自主循环，按需调工具 |
| **商品名确认** | 完整的 7 步子流程（提取→对齐→KG→确认） | 仅复用前 3 步（提取→对齐→确认） |
| **搜索策略** | 四路并行召回（全量执行） | 按需调用，最多 3 次 |
| **结果融合** | RRF 倒排秩融合 + BGE Reranker 精排 | 无融合层，LLM 自行综合 |
| **答案生成** | LLM 按固定 prompt 模板生成 | LLM 自主判断时机 + 组织语言 |
| **评估** | 外部调用 eval 脚本 | 内置实时 RAGAS 评估 |
| **调用链可见性** | 仅日志 | 前端 chip 可视化 + hover 详情 |
| **适用场景** | 高频稳定查询、对质量要求高 | 复杂模糊查询、需要灵活性 |
| **响应速度** | 较快（固定路径，无 LLM 决策开销） | 较慢（多次 LLM 调用决策） |

**复用关系**：Agentic RAG 的三个搜索工具全部复用传统 RAG 的节点函数，无重复实现。

---

## 核心亮点

### 1. 双向量检索
BGE-M3 单次推理同时产出 **1024 维稠密向量**（COSINE）和**变长稀疏向量**（IP），Milvus WeightedRanker 以 0.8/0.2 权重融合——语义匹配 + 关键词匹配互补。

### 2. 三级检索漏斗（传统模式）
```
多路召回 (Top 50) → RRF 融合 (Top 10) → Cross-Encoder 精排 (Top 5)
```

### 3. Agentic 自主决策
基于 LangGraph `create_react_agent`，LLM 根据搜索结果自主判断下一步（继续搜索 or 直接回答），支持工具输出截断（300 字/条）防止上下文溢出。

### 4. 主题识别预处理
Agent 循环前强制运行商品名识别，确保搜索范围准确，Agent 专注于策略选择而非命名实体识别。

### 5. 幂等设计
同一个商品重复导入时，先按 `item_name` 删旧数据再插入新数据，避免搜索结果出现新旧混合。

### 6. 图片可搜索化
PDF 中的图片经 Qwen3-VL 生成中文摘要后嵌入 Chunk，BGE-M3 向量化，用户搜"怎么接线"能命中接线图。

### 7. 容错降级
- LLM 超时 → file_title 兜底，不阻塞整条链路
- MinIO 上传失败 → 仅在日志告警，本地文件正常处理
- Agent 异常 → 返回错误信息，不影响服务

---

## 测试

```bash
# 单节点测试
uv run python -m app.import_process.agent.nodes.node_pdf_to_md

# API 接口测试
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/health

curl -X POST http://127.0.0.1:8001/query -H "Content-Type: application/json" -d '{"query":"MateBook B3-410的功能？","is_stream":false}'
curl -X POST http://127.0.0.1:8002/query -H "Content-Type: application/json" -d '{"query":"MateBook B3-410的功能？","is_stream":false}'
```

---

## 技术选型说明

| 选择 | 原因 |
|---|---|
| BGE-M3 vs text-embedding-3 | 单次推理双向量输出，1024 维比 1536 维存储更小，支持中英双语 |
| LangGraph vs Chain | 有向图支持条件分支 + 并行 fan-out，同时支持 Agentic create_react_agent |
| MinerU vs PyPDF2 | 复杂排版（双栏、表格、公式）PyPDF2 丢失率超 30%，MinerU 基于大模型准确还原 |
| Milvus vs FAISS | 分布式部署、双向量索引、自动扩缩容，FAISS 是单机库 |
| HNSW vs IVF | 查询延迟 2ms vs 10ms，适合低延迟 RAG 场景 |

---

## 许可证

MIT License
