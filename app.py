from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import uuid
import redis
import time
import os
from typing import Dict, List, Optional, Literal
from redis.connection import ConnectionPool, SSLConnection
import json

# ---------------------- 1. 初始化配置 ----------------------
app = FastAPI(title="GPT-Style API Controller", version="1.0")

# 跨域配置（适配外部调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Redis连接池（保留原有逻辑） ----------------------
def create_redis_pool() -> ConnectionPool:
    try:
        redis_host = os.getenv("REDIS_HOST")
        redis_port = int(os.getenv("REDIS_PORT", 6380))
        redis_password = os.getenv("REDIS_PASSWORD")
        
        if not all([redis_host, redis_password]):
            raise ValueError("REDIS_HOST和REDIS_PASSWORD必须配置")

        pool = ConnectionPool(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
            connection_class=SSLConnection,
            socket_timeout=5,
            retry_on_timeout=True,
            max_connections=10
        )
        
        with redis.Redis(connection_pool=pool) as client:
            client.ping()
        print("Redis连接池初始化成功")
        return pool
    except Exception as e:
        raise RuntimeError(f"Redis连接池初始化失败：{str(e)}")

redis_pool = create_redis_pool()

def get_redis_client() -> redis.Redis:
    with redis.Redis(connection_pool=redis_pool) as client:
        yield client

# ---------------------- 2. 标准GPT API鉴权（对齐OpenAI：Bearer Token） ----------------------
def verify_gpt_api_key(authorization: Optional[str] = Header(None)):
    # 标准GPT API用 Authorization: Bearer <API_KEY> 鉴权
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication. Use 'Authorization: Bearer <API_KEY>'",
            headers={"WWW-Authenticate": "Bearer"}
        )
    valid_key = os.getenv("CONTROLLER_API_KEY", "default-key")
    if authorization.split(" ")[1] != valid_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return authorization.split(" ")[1]

# ---------------------- 3. 处理节点配置（保留原有逻辑） ----------------------
PROCESSING_NODES = [
    {
        "url": os.getenv("NODE_1_URL", "https://fiewolf1000-bc_modle_space.hf.space"),
        "api_key": os.getenv("NODE_API_KEY", "node-key"),
        "health": "unknown",
        "last_check": 0,
        "load": 0
    },
    {
        "url": os.getenv("NODE_2_URL", "https://fiewolf1000-bc_modle_space1.hf.space"),
        "api_key": os.getenv("NODE_API_KEY", "node-key"),
        "health": "unknown",
        "last_check": 0,
        "load": 0
    }
]

# ---------------------- 4. 工具函数（保留原有逻辑，新增参数映射） ----------------------
async def check_node_health(node: Dict) -> bool:
    current_time = time.time()
    if current_time - node["last_check"] < 300:
        return node["health"] == "alive"
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{node['url']}/health",
                params={"api_key": node["api_key"]}
            )
        if resp.status_code == 200 and resp.json()["status"] == "alive":
            node["health"] = "alive"
            node["last_check"] = current_time
            return True
        node["health"] = "dead"
        return False
    except Exception as e:
        node["health"] = "dead"
        return False

def get_least_loaded_node() -> Optional[Dict]:
    healthy_nodes = [n for n in PROCESSING_NODES if check_node_health(n)]
    if not healthy_nodes:
        return None
    return min(healthy_nodes, key=lambda x: x["load"])

def task_key(user_id: str, task_id: str) -> str:
    return f"user:{user_id}:task:{task_id}"

# 生成标准GPT API的ID（格式：cmpl-xxx 或 chatcmpl-xxx）
def generate_gpt_id(type: Literal["completion", "chat"]) -> str:
    prefix = "cmpl-" if type == "completion" else "chatcmpl-"
    return prefix + uuid.uuid4().hex[:24]

# ---------------------- 5. 标准GPT API请求/响应模型（核心适配） ----------------------
# 5.1 文本补全接口（/v1/completions）模型
class CompletionRequest(BaseModel):
    model: str = Field(default="gpt-3.5-turbo-instruct", description="模型名（仅做标识，实际用处理节点模型）")
    prompt: str = Field(description="生成文本的提示词")
    max_tokens: int = Field(default=512, ge=1, le=2048, description="最大生成token数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="随机性（0=确定性，2=最大随机性）")
    top_p: float = Field(default=1.0, description="核采样（与temperature二选一）")  # 简化：暂不实际使用
    n: int = Field(default=1, ge=1, le=1, description="生成结果数（暂仅支持1）")
    stop: Optional[List[str]] = Field(default=None, description="停止符")  # 简化：暂不实际使用
    user: Optional[str] = Field(default="default_user", description="用户ID（用于关联对话历史）")

class CompletionChoice(BaseModel):
    text: str = Field(description="生成的文本")
    index: int = Field(default=0, description="结果索引")
    finish_reason: Optional[str] = Field(default="stop", description="结束原因")
    logprobs: Optional[Dict] = Field(default=None, description="概率信息（简化：暂不返回）")

class CompletionUsage(BaseModel):
    prompt_tokens: int = Field(description="提示词token数（估算）")
    completion_tokens: int = Field(description="生成文本token数（估算）")
    total_tokens: int = Field(description="总token数")

class CompletionResponse(BaseModel):
    id: str = Field(description="请求ID（标准GPT格式）")
    object: str = Field(default="text_completion", description="对象类型")
    created: int = Field(description="创建时间戳（秒）")
    model: str = Field(description="使用的模型名")
    choices: List[CompletionChoice] = Field(description="生成结果列表")
    usage: CompletionUsage = Field(description="token使用统计")

# 5.2 对话补全接口（/v1/chat/completions）模型
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(description="角色（用户/助手/系统）")
    content: str = Field(description="消息内容")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-3.5-turbo", description="模型名（仅做标识）")
    messages: List[ChatMessage] = Field(description="对话历史（包含system/user/assistant）")
    max_tokens: int = Field(default=512, ge=1, le=2048, description="最大生成token数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="随机性")
    user: Optional[str] = Field(default="default_user", description="用户ID")

class ChatCompletionChoice(BaseModel):
    message: ChatMessage = Field(description="生成的消息（assistant角色）")
    index: int = Field(default=0, description="结果索引")
    finish_reason: Optional[str] = Field(default="stop", description="结束原因")

class ChatCompletionResponse(BaseModel):
    id: str = Field(description="请求ID（标准GPT格式：chatcmpl-xxx）")
    object: str = Field(default="chat.completion", description="对象类型")
    created: int = Field(description="创建时间戳（秒）")
    model: str = Field(description="使用的模型名")
    choices: List[ChatCompletionChoice] = Field(description="生成结果列表")
    usage: CompletionUsage = Field(description="token使用统计")

# ---------------------- 6. 标准GPT API核心接口（新增） ----------------------
# 6.1 文本补全接口（/v1/completions）- 对齐GPT-3
@app.post("/v1/completions", response_model=CompletionResponse)
async def gpt_completions(
    req: CompletionRequest,
    api_key: str = Depends(verify_gpt_api_key),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    # 1. 基础校验
    if req.n != 1:
        raise HTTPException(
            status_code=400,
            detail="Parameter 'n' only supports 1 (temporarily)"
        )
    
    # 2. 分配处理节点
    node = get_least_loaded_node()
    if not node:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: No healthy processing nodes"
        )
    
    # 3. 生成任务ID（关联用户对话历史）
    task_id = generate_gpt_id("completion").replace("cmpl-", "task-")
    user_id = req.user or "default_user"
    key = task_key(user_id, task_id)
    
    # 4. 调用处理节点生成文本
    try:
        node["load"] += 1
        async with httpx.AsyncClient(timeout=60) as client:
            # 适配处理节点的/generate/code接口（将prompt传递，忽略节点返回的"code"命名）
            node_resp = await client.post(
                f"{node['url']}/generate/code",
                json={
                    "prompt": req.prompt,
                    "max_new_tokens": req.max_tokens,
                    "temperature": req.temperature
                },
                params={"api_key": node["api_key"]}
            )
        node["load"] -= 1
        node_resp.raise_for_status()
        node_result = node_resp.json()
    except Exception as e:
        node["load"] -= 1
        raise HTTPException(
            status_code=500,
            detail=f"Processing node error: {str(e)}"
        )
    
    # 5. 存储对话历史到Redis
    history = [{"role": "user", "content": req.prompt}, {"role": "assistant", "content": node_result["code"]}]
    redis_client.hset(key, mapping={
        "task_id": task_id,
        "user_id": user_id,
        "model": req.model,
        "created": str(int(time.time())),
        "updated": str(int(time.time())),
        "history": json.dumps(history)
    })
    redis_client.sadd(f"user:{user_id}:tasks", task_id)
    
    # 6. 估算token使用量（简化：按字符数粗略估算，1token≈4字符）
    prompt_tokens = len(req.prompt) // 4
    completion_tokens = len(node_result["code"]) // 4
    total_tokens = prompt_tokens + completion_tokens
    
    # 7. 生成标准GPT响应
    return CompletionResponse(
        id=generate_gpt_id("completion"),
        created=int(time.time()),
        model=req.model,
        choices=[CompletionChoice(
            text=node_result["code"],  # 节点返回的"code"实际是生成文本，直接映射为text
            index=0,
            finish_reason="stop"
        )],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
    )

# 6.2 对话补全接口（/v1/chat/completions）- 对齐GPT-3.5/4
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def gpt_chat_completions(
    req: ChatCompletionRequest,
    api_key: str = Depends(verify_gpt_api_key),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    # 1. 构建对话上下文（拼接system/user/assistant历史）
    context = ""
    for msg in req.messages:
        context += f"{msg.role}: {msg.content}\n"
    # 补充assistant前缀（提示节点生成助手回复）
    context += "assistant: "
    
    # 2. 分配处理节点
    node = get_least_loaded_node()
    if not node:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: No healthy processing nodes"
        )
    
    # 3. 生成任务ID（关联用户对话历史）
    task_id = generate_gpt_id("chat").replace("chatcmpl-", "task-")
    user_id = req.user or "default_user"
    key = task_key(user_id, task_id)
    
    # 4. 调用处理节点生成对话回复
    try:
        node["load"] += 1
        async with httpx.AsyncClient(timeout=60) as client:
            node_resp = await client.post(
                f"{node['url']}/generate/code",
                json={
                    "prompt": context,
                    "max_new_tokens": req.max_tokens,
                    "temperature": req.temperature
                },
                params={"api_key": node["api_key"]}
            )
        node["load"] -= 1
        node_resp.raise_for_status()
        node_result = node_resp.json()
        assistant_content = node_result["code"].strip()  # 节点返回的"code"即助手回复
    except Exception as e:
        node["load"] -= 1
        raise HTTPException(
            status_code=500,
            detail=f"Processing node error: {str(e)}"
        )
    
    # 5. 存储完整对话历史到Redis
    req.messages.append({"role": "assistant", "content": assistant_content})
    redis_client.hset(key, mapping={
        "task_id": task_id,
        "user_id": user_id,
        "model": req.model,
        "created": str(int(time.time())),
        "updated": str(int(time.time())),
        "history": json.dumps([msg.dict() for msg in req.messages])
    })
    redis_client.sadd(f"user:{user_id}:tasks", task_id)
    
    # 6. 估算token使用量
    prompt_text = "\n".join([f"{msg.role}: {msg.content}" for msg in req.messages[:-1]])  # 排除新增的assistant
    prompt_tokens = len(prompt_text) // 4
    completion_tokens = len(assistant_content) // 4
    total_tokens = prompt_tokens + completion_tokens
    
    # 7. 生成标准GPT对话响应
    return ChatCompletionResponse(
        id=generate_gpt_id("chat"),
        created=int(time.time()),
        model=req.model,
        choices=[ChatCompletionChoice(
            message=ChatMessage(
                role="assistant",
                content=assistant_content
            ),
            index=0,
            finish_reason="stop"
        )],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
    )

# ---------------------- 7. 保留原有管理接口（供内部使用） ----------------------
@app.get("/nodes/health")
async def node_health(api_key: str = Depends(verify_gpt_api_key)):
    status = []
    for node in PROCESSING_NODES:
        await check_node_health(node)
        status.append({
            "url": node["url"],
            "health": node["health"],
            "load": node["load"],
            "last_check": time.ctime(node["last_check"]) if node["last_check"] else "Never"
        })
    return {"status": "success", "nodes": status}

@app.get("/user/tasks/{user_id}")
async def list_tasks(
    user_id: str,
    api_key: str = Depends(verify_gpt_api_key),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    task_ids = redis_client.smembers(f"user:{user_id}:tasks")
    tasks = []
    for tid in task_ids:
        data = redis_client.hgetall(task_key(user_id, tid))
        if data:
            tasks.append({
                "task_id": tid,
                "model": data.get("model", "unknown"),
                "created": time.ctime(int(data["created"])),
                "updated": time.ctime(int(data["updated"])),
                "history_length": len(json.loads(data["history"]))
            })
    return {"status": "success", "tasks": tasks}
