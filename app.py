from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import uuid
import redis
import time
import os
import logging
from typing import Dict, List, Optional, Literal
from redis.connection import ConnectionPool, SSLConnection
import json

# ---------------------- 1. 初始化日志配置（核心新增：详细日志输出） ----------------------
# 配置日志格式：时间戳 + 日志级别 + 模块 + 消息
logging.basicConfig(
    level=logging.INFO,  # 日志级别：INFO（常规信息）、ERROR（错误）、DEBUG（调试，可选）
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"  # 时间格式
)
logger = logging.getLogger(__name__)  # 创建日志实例，后续用logger输出日志

# ---------------------- 2. 初始化配置 ----------------------
app = FastAPI(title="GPT-Style API Controller", version="1.0")

# 跨域配置（适配外部调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- 3. Redis连接池（新增日志：连接过程、成功/失败） ----------------------
def create_redis_pool() -> ConnectionPool:
    try:
        redis_host = os.getenv("REDIS_HOST")
        redis_port = int(os.getenv("REDIS_PORT", 6380))
        redis_password = os.getenv("REDIS_PASSWORD")
        
        # 日志：打印读取到的配置（隐藏密码，避免泄露）
        logger.info(f"读取Redis配置：host={redis_host}, port={redis_port}, password=******")
        
        if not all([redis_host, redis_password]):
            logger.error("Redis配置不完整：REDIS_HOST或REDIS_PASSWORD未设置")
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
        
        # 验证连接
        with redis.Redis(connection_pool=pool) as client:
            client.ping()
        logger.info("Redis连接池初始化成功，已验证连接可用性")
        return pool
    except Exception as e:
        logger.error(f"Redis连接池初始化失败：{str(e)}", exc_info=True)  # exc_info=True：打印完整异常栈
        raise RuntimeError(f"Redis连接池初始化失败：{str(e)}")

# 初始化连接池（日志会在create_redis_pool中输出）
redis_pool = create_redis_pool()

def get_redis_client() -> redis.Redis:
    with redis.Redis(connection_pool=redis_pool) as client:
        yield client

# ---------------------- 4. 标准GPT API鉴权（新增日志：鉴权结果、错误详情） ----------------------
def verify_gpt_api_key(authorization: Optional[str] = Header(None)):
    try:
        # 日志：记录鉴权请求（隐藏完整token，仅显示前6位）
        auth_token = authorization.split(" ")[1] if (authorization and authorization.startswith("Bearer ")) else "None"
        logger.info(f"收到API鉴权请求：token_prefix={auth_token[:6]}***")
        
        if not authorization or not authorization.startswith("Bearer "):
            logger.warning("鉴权失败：缺少Authorization头或格式错误（需为Bearer Token）")
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication. Use 'Authorization: Bearer <API_KEY>'",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        valid_key = os.getenv("CONTROLLER_API_KEY", "default-key")
        request_key = authorization.split(" ")[1]
        if request_key != valid_key:
            logger.warning(f"鉴权失败：API密钥不匹配（请求密钥前缀：{request_key[:6]}***）")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        logger.info(f"鉴权成功：token_prefix={request_key[:6]}***")
        return request_key
    except HTTPException:
        raise  # 重新抛出HTTPException，不捕获
    except Exception as e:
        logger.error(f"鉴权过程异常：{str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Authentication process error")

# ---------------------- 5. 处理节点配置 ----------------------
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

# ---------------------- 6. 工具函数（新增日志：节点健康检查结果、负载均衡选择） ----------------------
async def check_node_health(node: Dict) -> bool:
    node_url = node["url"]
    current_time = time.time()
    
    # 日志：记录健康检查触发原因（缓存过期/首次检查）
    if current_time - node["last_check"] < 300:
        logger.debug(f"节点{node_url}：健康检查结果缓存未过期（剩余{300 - (current_time - node['last_check']):.0f}秒），直接返回缓存状态{node['health']}")
        return node["health"] == "alive"
    
    logger.info(f"开始检查节点健康状态：url={node_url}，上次检查时间={time.ctime(node['last_check'])}")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{node_url}/health",
                params={"api_key": node["api_key"]}
            )
        
        # 日志：记录节点返回的状态码和响应
        logger.debug(f"节点{node_url}健康检查响应：status_code={resp.status_code}，response={resp.text[:100]}***")
        
        if resp.status_code == 200 and resp.json()["status"] == "alive":
            node["health"] = "alive"
            node["last_check"] = current_time
            logger.info(f"节点{node_url}健康检查通过，状态更新为alive")
            return True
        else:
            node["health"] = "dead"
            logger.warning(f"节点{node_url}健康检查失败：状态码{resp.status_code}，响应内容{resp.text[:100]}***")
            return False
    except Exception as e:
        node["health"] = "dead"
        logger.error(f"节点{node_url}健康检查抛出异常：{str(e)}", exc_info=True)
        return False

def get_least_loaded_node() -> Optional[Dict]:
    # 先过滤健康节点，日志记录过滤结果
    healthy_nodes = [n for n in PROCESSING_NODES if check_node_health(n)]
    logger.info(f"负载均衡：当前健康节点数量={len(healthy_nodes)}，总节点数量={len(PROCESSING_NODES)}")
    
    if not healthy_nodes:
        logger.error("负载均衡：无健康节点可用，无法分配任务")
        return None
    
    # 日志：打印所有健康节点的当前负载
    for node in healthy_nodes:
        logger.debug(f"健康节点{node['url']}：当前负载={node['load']}")
    
    # 选择负载最低的节点
    selected_node = min(healthy_nodes, key=lambda x: x["load"])
    logger.info(f"负载均衡：选择节点{selected_node['url']}（当前负载={selected_node['load']}）")
    return selected_node

def task_key(user_id: str, task_id: str) -> str:
    return f"user:{user_id}:task:{task_id}"

def generate_gpt_id(type: Literal["completion", "chat"]) -> str:
    prefix = "cmpl-" if type == "completion" else "chatcmpl-"
    gpt_id = prefix + uuid.uuid4().hex[:24]
    logger.debug(f"生成GPT风格请求ID：type={type}，id={gpt_id}")
    return gpt_id

# ---------------------- 7. 标准GPT API请求/响应模型（无修改） ----------------------
class CompletionRequest(BaseModel):
    model: str = Field(default="gpt-3.5-turbo-instruct", description="模型名（仅做标识）")
    prompt: str = Field(description="生成文本的提示词")
    max_tokens: int = Field(default=512, ge=1, le=2048, description="最大生成token数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="随机性")
    top_p: float = Field(default=1.0, description="核采样（暂不使用）")
    n: int = Field(default=1, ge=1, le=1, description="生成结果数（暂仅支持1）")
    stop: Optional[List[str]] = Field(default=None, description="停止符（暂不使用）")
    user: Optional[str] = Field(default="default_user", description="用户ID")

class CompletionChoice(BaseModel):
    text: str = Field(description="生成的文本")
    index: int = Field(default=0, description="结果索引")
    finish_reason: Optional[str] = Field(default="stop", description="结束原因")
    logprobs: Optional[Dict] = Field(default=None, description="概率信息（暂不返回）")

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

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(description="角色")
    content: str = Field(description="消息内容")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-3.5-turbo", description="模型名（仅做标识）")
    messages: List[ChatMessage] = Field(description="对话历史")
    max_tokens: int = Field(default=512, ge=1, le=2048, description="最大生成token数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="随机性")
    user: Optional[str] = Field(default="default_user", description="用户ID")

class ChatCompletionChoice(BaseModel):
    message: ChatMessage = Field(description="生成的消息（assistant角色）")
    index: int = Field(default=0, description="结果索引")
    finish_reason: Optional[str] = Field(default="stop", description="结束原因")

class ChatCompletionResponse(BaseModel):
    id: str = Field(description="请求ID（标准GPT格式）")
    object: str = Field(default="chat.completion", description="对象类型")
    created: int = Field(description="创建时间戳（秒）")
    model: str = Field(description="使用的模型名")
    choices: List[ChatCompletionChoice] = Field(description="生成结果列表")
    usage: CompletionUsage = Field(description="token使用统计")

# ---------------------- 8. 标准GPT API核心接口（新增日志：请求参数、节点调用结果、错误详情） ----------------------
@app.post("/v1/completions", response_model=CompletionResponse)
async def gpt_completions(
    req: CompletionRequest,
    api_key: str = Depends(verify_gpt_api_key),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    # 日志：记录完整请求参数（隐藏用户ID，可选）
    logger.info(f"收到/v1/completions请求：user={req.user[:6]}***, model={req.model}, max_tokens={req.max_tokens}, prompt_length={len(req.prompt)}")
    
    try:
        # 1. 基础校验
        if req.n != 1:
            logger.warning(f"请求参数校验失败：n={req.n}（暂仅支持1）")
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
        
        # 3. 生成任务ID
        task_id = generate_gpt_id("completion").replace("cmpl-", "task-")
        user_id = req.user or "default_user"
        key = task_key(user_id, task_id)
        logger.debug(f"创建任务：task_id={task_id}, user_id={user_id}, redis_key={key}")
        
        # 4. 调用处理节点生成文本
        logger.info(f"调用处理节点生成文本：node_url={node['url']}, task_id={task_id}, max_new_tokens={req.max_tokens}")
        try:
            node["load"] += 1
            logger.debug(f"更新节点负载：node_url={node['url']}，负载从{node['load']-1}增至{node['load']}")
            
            async with httpx.AsyncClient(timeout=60) as client:
                node_resp = await client.post(
                    f"{node['url']}/generate/code",
                    json={
                        "prompt": req.prompt,
                        "max_new_tokens": req.max_tokens,
                        "temperature": req.temperature
                    },
                    params={"api_key": node["api_key"]}
                )
            
            # 日志：记录节点响应状态和结果长度
            node_result = node_resp.json()
            logger.debug(f"节点返回结果：node_url={node['url']}, status_code={node_resp.status_code}, result_length={len(node_result.get('code', ''))}")
            
            node["load"] -= 1
            logger.debug(f"更新节点负载：node_url={node['url']}，负载从{node['load']+1}降至{node['load']}")
            node_resp.raise_for_status()  # 触发HTTP错误（如4xx/5xx）
        except Exception as e:
            node["load"] -= 1
            logger.debug(f"节点调用失败后更新负载：node_url={node['url']}，负载从{node['load']+1}降至{node['load']}")
            logger.error(f"处理节点调用失败：node_url={node['url']}, task_id={task_id}, error={str(e)}", exc_info=True)
            raise
        
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
        logger.info(f"任务历史存储成功：redis_key={key}, history_length={len(history)}")
        
        # 6. 估算token使用量
        prompt_tokens = len(req.prompt) // 4
        completion_tokens = len(node_result["code"]) // 4
        total_tokens = prompt_tokens + completion_tokens
        logger.debug(f"Token使用估算：prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}")
        
        # 7. 生成响应
        gpt_id = generate_gpt_id("completion")
        logger.info(f"/v1/completions请求处理完成：request_id={gpt_id}, task_id={task_id}, user_id={user_id}")
        return CompletionResponse(
            id=gpt_id,
            created=int(time.time()),
            model=req.model,
            choices=[CompletionChoice(
                text=node_result["code"],
                index=0,
                finish_reason="stop"
            )],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
    except HTTPException as e:
        # 日志：记录HTTP错误（如400/401/503）
        logger.error(f"/v1/completions请求返回HTTP错误：status_code={e.status_code}, detail={e.detail}")
        raise
    except Exception as e:
        # 日志：记录未捕获的异常
        logger.error(f"/v1/completions请求处理异常：{str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def gpt_chat_completions(
    req: ChatCompletionRequest,
    api_key: str = Depends(verify_gpt_api_key),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    # 日志：记录对话请求的基本信息（消息数量、用户ID）
    user_id = req.user or "default_user"
    logger.info(f"收到/v1/chat/completions请求：user={user_id[:6]}***, model={req.model}, max_tokens={req.max_tokens}, message_count={len(req.messages)}")
    
    try:
        # 1. 构建对话上下文
        context = ""
        for msg in req.messages:
            context += f"{msg.role}: {msg.content}\n"
            logger.debug(f"对话历史：role={msg.role}, content_length={len(msg.content)}")
        context += "assistant: "
        logger.debug(f"构建对话上下文：length={len(context)}")
        
        # 2. 分配处理节点
        node = get_least_loaded_node()
        if not node:
            raise HTTPException(
                status_code=503,
                detail="Service unavailable: No healthy processing nodes"
            )
        
        # 3. 生成任务ID
        task_id = generate_gpt_id("chat").replace("chatcmpl-", "task-")
        key = task_key(user_id, task_id)
        logger.debug(f"创建对话任务：task_id={task_id}, user_id={user_id}, redis_key={key}")
        
        # 4. 调用处理节点生成回复
        logger.info(f"调用处理节点生成对话回复：node_url={node['url']}, task_id={task_id}")
        try:
            node["load"] += 1
            logger.debug(f"更新节点负载：node_url={node['url']}，负载从{node['load']-1}增至{node['load']}")
            
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
            
            node_result = node_resp.json()
            assistant_content = node_result["code"].strip()
            logger.debug(f"节点返回对话回复：length={len(assistant_content)}, status_code={node_resp.status_code}")
            
            node["load"] -= 1
            logger.debug(f"更新节点负载：node_url={node['url']}，负载从{node['load']+1}降至{node['load']}")
            node_resp.raise_for_status()
        except Exception as e:
            node["load"] -= 1
            logger.debug(f"节点调用失败后更新负载：node_url={node['url']}，负载从{node['load']+1}降至{node['load']}")
            logger.error(f"处理节点调用失败：node_url={node['url']}, task_id={task_id}, error={str(e)}", exc_info=True)
            raise
        
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
        logger.info(f"对话历史存储成功：redis_key={key}, message_count={len(req.messages)}")
        
        # 6. 估算token使用量
        prompt_text = "\n".join([f"{msg.role}: {msg.content}" for msg in req.messages[:-1]])
        prompt_tokens = len(prompt_text) // 4
        completion_tokens = len(assistant_content) // 4
        total_tokens = prompt_tokens + completion_tokens
        logger.debug(f"Token使用估算：prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}")
        
        # 7. 生成响应
        gpt_id = generate_gpt_id("chat")
        logger.info(f"/v1/chat/completions请求处理完成：request_id={gpt_id}, task_id={task_id}, user_id={user_id}")
        return ChatCompletionResponse(
            id=gpt_id,
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
    except HTTPException as e:
        logger.error(f"/v1/chat/completions请求返回HTTP错误：status_code={e.status_code}, detail={e.detail}")
        raise
    except Exception as e:
        logger.error(f"/v1/chat/completions请求处理异常：{str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# ---------------------- 9. 管理接口（新增日志：节点状态、用户任务列表） ----------------------
@app.get("/nodes/health")
async def node_health(api_key: str = Depends(verify_gpt_api_key)):
    logger.info("收到节点健康状态查询请求")
    status = []
    for node in PROCESSING_NODES:
        is_alive = await check_node_health(node)
        status.append({
            "url": node["url"],
            "health": node["health"],
            "load": node["load"],
            "last_check": time.ctime(node["last_check"]) if node["last_check"] else "Never"
        })
    logger.debug(f"节点健康状态查询结果：{json.dumps(status, indent=2)}")
    return {"status": "success", "nodes": status}

@app.get("/user/tasks/{user_id}")
async def list_tasks(
    user_id: str,
    api_key: str = Depends(verify_gpt_api_key),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    logger.info(f"收到用户任务列表查询请求：user_id={user_id}")
    try:
        task_ids = redis_client.smembers(f"user:{user_id}:tasks")
        logger.debug(f"用户{user_id}的任务数量：{len(task_ids)}")
        
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
        logger.info(f"用户{user_id}任务列表查询完成：返回任务数量={len(tasks)}")
        return {"status": "success", "tasks": tasks}
    except Exception as e:
        logger.error(f"用户任务列表查询异常：user_id={user_id}, error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch user tasks")
