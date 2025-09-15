from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uuid
import redis
import time
import os
from typing import Dict, List, Optional
from redis.connection import ConnectionPool, SSLConnection  # 引入SSL连接类

# ---------------------- 1. 初始化配置 ----------------------
app = FastAPI(title="Distributed Agent Controller", version="1.0")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- 修复Redis连接池（兼容低版本redis-py） ----------------------
def create_redis_pool() -> ConnectionPool:
    try:
        # 从环境变量获取配置（必须在Space中正确设置）
        redis_host = os.getenv("REDIS_HOST")
        redis_port = int(os.getenv("REDIS_PORT", 6380))
        redis_password = os.getenv("REDIS_PASSWORD")
        
        if not all([redis_host, redis_password]):
            raise ValueError("REDIS_HOST和REDIS_PASSWORD必须配置")

        # 关键修复：使用SSLConnection类，替代ssl=True参数（兼容低版本redis-py）
        pool = ConnectionPool(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
            connection_class=SSLConnection,  # 强制使用SSL加密连接
            socket_timeout=5,
            retry_on_timeout=True,
            max_connections=10
        )
        
        # 验证连接
        with redis.Redis(connection_pool=pool) as client:
            client.ping()  # 成功会返回True
        print("Redis连接池初始化成功")
        return pool
    except Exception as e:
        raise RuntimeError(f"Redis连接池初始化失败：{str(e)}")

# 初始化连接池
redis_pool = create_redis_pool()

# 依赖注入：获取Redis客户端
def get_redis_client() -> redis.Redis:
    with redis.Redis(connection_pool=redis_pool) as client:
        yield client

# ---------------------- 2. API密钥验证 ----------------------
def verify_api_key(api_key: Optional[str] = None):
    valid_key = os.getenv("CONTROLLER_API_KEY", "default-key")
    if api_key != valid_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# ---------------------- 3. 处理节点配置 ----------------------
PROCESSING_NODES = [
    {
        "url": os.getenv("NODE_1_URL", "https://your-username-node1.hf.space"),
        "api_key": os.getenv("NODE_API_KEY", "node-key"),
        "health": "unknown",
        "last_check": 0,
        "load": 0
    },
    {
        "url": os.getenv("NODE_2_URL", "https://your-username-node2.hf.space"),
        "api_key": os.getenv("NODE_API_KEY", "node-key"),
        "health": "unknown",
        "last_check": 0,
        "load": 0
    }
]

# ---------------------- 4. 数据模型 ----------------------
class CreateTaskRequest(BaseModel):
    user_id: str
    task_name: str
    initial_prompt: str

class SendMessageRequest(BaseModel):
    user_id: str
    task_id: str
    message: str
    max_new_tokens: int = 512
    temperature: float = 0.7

# ---------------------- 5. 工具函数 ----------------------
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

# ---------------------- 6. 核心接口 ----------------------
@app.post("/task/create")
async def create_task(
    req: CreateTaskRequest,
    api_key: str = Depends(verify_api_key),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    task_data = {
        "task_id": task_id,
        "name": req.task_name,
        "created": str(int(time.time())),
        "updated": str(int(time.time())),
        "history": '[{"role": "user", "content": "%s"}]' % req.initial_prompt.replace('"', '\\"')
    }

    redis_client.hset(task_key(req.user_id, task_id), mapping=task_data)
    redis_client.sadd(f"user:{req.user_id}:tasks", task_id)

    node = get_least_loaded_node()
    if not node:
        raise HTTPException(status_code=503, detail="No available nodes")

    try:
        node["load"] += 1
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{node['url']}/generate/code",
                json={"prompt": req.initial_prompt, "max_new_tokens": 512},
                params={"api_key": node["api_key"]}
            )
        node["load"] -= 1
        resp.raise_for_status()
    except Exception as e:
        node["load"] -= 1
        raise HTTPException(status_code=500, detail=f"Node error: {str(e)}")

    import json
    result = resp.json()
    history = json.loads(task_data["history"])
    history.append({"role": "assistant", "content": result["code"].replace('"', '\\"')})
    redis_client.hset(
        task_key(req.user_id, task_id),
        mapping={
            "history": json.dumps(history),
            "updated": str(int(time.time()))
        }
    )

    return {
        "status": "success",
        "task_id": task_id,
        "response": result["code"]
    }

@app.post("/task/message")
async def send_message(
    req: SendMessageRequest,
    api_key: str = Depends(verify_api_key),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    import json
    key = task_key(req.user_id, req.task_id)
    if not redis_client.exists(key):
        raise HTTPException(status_code=404, detail="Task not found")

    task_data = redis_client.hgetall(key)
    history = json.loads(task_data["history"])
    history.append({"role": "user", "content": req.message.replace('"', '\\"')})

    context = "\n".join([f"{m['role']}: {m['content']}" for m in history])

    node = get_least_loaded_node()
    if not node:
        raise HTTPException(status_code=503, detail="No available nodes")

    try:
        node["load"] += 1
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{node['url']}/generate/code",
                json={
                    "prompt": context,
                    "max_new_tokens": req.max_new_tokens,
                    "temperature": req.temperature
                },
                params={"api_key": node["api_key"]}
            )
        node["load"] -= 1
        resp.raise_for_status()
    except Exception as e:
        node["load"] -= 1
        raise HTTPException(status_code=500, detail=f"Node error: {str(e)}")

    result = resp.json()
    history.append({"role": "assistant", "content": result["code"].replace('"', '\\"')})
    redis_client.hset(
        key,
        mapping={
            "history": json.dumps(history),
            "updated": str(int(time.time()))
        }
    )

    return {"status": "success", "response": result["code"]}

@app.get("/user/tasks/{user_id}")
async def list_tasks(
    user_id: str,
    api_key: str = Depends(verify_api_key),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    import json
    task_ids = redis_client.smembers(f"user:{user_id}:tasks")
    tasks = []
    for tid in task_ids:
        data = redis_client.hgetall(task_key(user_id, tid))
        if data:
            tasks.append({
                "task_id": tid,
                "name": data["name"],
                "created": time.ctime(int(data["created"])),
                "updated": time.ctime(int(data["updated"])),
                "history_length": len(json.loads(data["history"]))
            })
    return {"status": "success", "tasks": tasks}

@app.get("/nodes/health")
async def node_health(api_key: str = Depends(verify_api_key)):
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