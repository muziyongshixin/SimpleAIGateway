import asyncio
import json
import json5
import copy
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any
import traceback
import aiohttp
import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("model_gateway.log"), logging.StreamHandler()],
)
logger = logging.getLogger("model_gateway")

http_proxy = os.getenv("http_proxy")
logger.info(f'Using http_proxy = {http_proxy}')
TIMEOUT = 120

# 定义模型配置项
class ModelInstance(BaseModel):
    url: str
    api_key: Optional[str] = None
    model_name: str
    weight: int = 1 # 权重，用于负载均衡

# 模型网关配置
class GatewayConfig(BaseModel):
    instances: List[ModelInstance]
    fallback_instances: List[ModelInstance]
    data_dir: str = "./data"
    # "round_robin" 或 "least_connections"
    load_balancing_strategy: str = "round_robin"
    error_threshold: int = 5 # 错误阈值，超过此值会触发告警
    error_window: int = 60 # 错误计算窗口（秒）
    alert_cooldown: int = 300 # 告警冷却时间（秒）

# 创建FastAPI应用
app = FastAPI(title="LLM Model Gateway")

# 全局变量
config: Optional[GatewayConfig] = None
config_file_path = "config.json5"
active_connections = defaultdict(int) # 实例ID -> 当前连接数
error_counters = defaultdict(lambda: deque(maxlen=100)) # 实例ID -> 最近错误时间列表
last_alert_time = defaultdict(float) # 实例ID -> 上次告警时间
current_instance_index = 0 # 用于轮询负载均衡
start_time = 0

# 新增：定义一个全局异步锁，保证文件写入不冲突
save_file_lock = asyncio.Lock()

# 初始化数据目录
def init_data_directories(data_dir: str):
    Path(data_dir).mkdir(exist_ok=True)
    Path(f"{data_dir}/requests_responses").mkdir(exist_ok=True)
    Path(f"{data_dir}/errors").mkdir(exist_ok=True)

# 载入配置文件
def load_config():
    global config, current_instance_index
    try:
        with open(config_file_path, "r") as f:
            config_data = json5.load(f)
        config = GatewayConfig(**config_data)
        init_data_directories(config.data_dir)
        current_instance_index = 0
        logger.info(f"配置已加载: {len(config.instances)} 个模型实例, 成功加载{len(config.fallback_instances)}个兜底实例")
        return True
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return False

# 配置文件监视器
class ConfigFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(config_file_path):
            logger.info("配置文件已更改，正在重新加载...")
            load_config()

# 选择模型实例 - 负载均衡
def select_instance(instance_list: List[ModelInstance]):
    global current_instance_index
    if not instance_list:
        raise HTTPException(status_code=500, detail="没有可用的模型实例")

    # 移除短时间内频繁出错的实例
    available_instances = []
    for idx, instance in enumerate(instance_list):
        instance_id = f"{instance.url}_{instance.model_name}"
        errors = error_counters[instance_id]
        recent_errors = sum(1 for t in errors if time.time() - t < config.error_window)
        if recent_errors < config.error_threshold:
            available_instances.append((idx, instance))

    if not available_instances:
        available_instances = [(idx, instance) for idx, instance in enumerate(instance_list)]
        logger.warning(f"所有实例都达到错误阈值，使用所有实例进行重试， 总实例数量 {len(available_instances)}")

    if config.load_balancing_strategy == "least_connections":
        available_instances.sort(key=lambda x: active_connections[f"{x[1].model_name}_{x[0]}"] / x[1].weight if x[1].weight > 0 else active_connections[f"{x[1].model_name}_{x[0]}"])
        selected_idx, selected_instance = available_instances[0]
    else:
        # 轮询策略（加权）
        weights = []
        for idx, instance in available_instances:
            weights.extend([idx] * instance.weight)
        if not weights:
            selected_idx, selected_instance = available_instances[0]
        else:
            current_instance_index = (current_instance_index + 1) % len(weights)
            selected_idx = weights[current_instance_index]
            selected_instance = instance_list[selected_idx]

    instance_id = f"{selected_instance.url}_{selected_instance.model_name}"
    logger.debug(f"选择实例 {instance_id}，当前连接数: {active_connections[instance_id]}")
    return selected_instance

# 记录错误并检查是否需要告警
def record_error(instance_id: str, error_msg: str):
    error_counters[instance_id].append(time.time())
    recent_errors = sum(1 for t in error_counters[instance_id] if time.time() - t < config.error_window)
    if (recent_errors >= config.error_threshold and time.time() - last_alert_time[instance_id] > config.alert_cooldown):
        last_alert_time[instance_id] = time.time()
        alert_message = f"实例 {instance_id} 在过去 {config.error_window} 秒内出现 {recent_errors} 次错误"
        logger.error(alert_message)
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        error_file = f"{config.data_dir}/errors/{timestamp}.log"
        with open(error_file, "a+") as f:
            f.writelines(f"时间: {datetime.now().isoformat()}\n 实例: {instance_id}\n 错误: {error_msg}\n 最近错误次数: {recent_errors}\n")
        # TODO 这里可以添加发送微信或者报警短信的逻辑

# 数据持久化，修改为每小时保存到一个文件中，每条记录按 JSON Lines 格式存储
async def save_request_response(request_id: str, model: str, request_data: dict, response_data: Any):
    try:
        # 如果是流式输出，则合并所有 token 的 content 到一行，不包含 id, object, created 等字段
        messages = request_data.get("messages", [])
        if request_data.get("stream", False):
            aggregated_content = ""
            for chunk in response_data:
                try:
                    content_piece = chunk.get("choices", [])[0].get("delta", {}).get("content", "")
                except Exception:
                    content_piece = ""
                aggregated_content += content_piece
            messages.append({'role': 'assistant', 'content': aggregated_content})
        else:
            assistant_message = response_data.get("choices", [])[0].get("message", {})
            messages.append(assistant_message)

        save_data = copy.copy(request_data)
        save_data.update({
            "_id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "messages": messages,
        })

        if messages[-1]['content'] is not None and len(messages[-1]['content'])>0:
            # 根据当前时间生成小时级的文件名（JSON Lines 格式）
            hour_filename = datetime.now().strftime("%Y%m%d_%H") + ".jsonl"
            response_file = f"{config.data_dir}/requests_responses/{hour_filename}"

            # 使用全局锁确保写入不冲突
            async with save_file_lock:
                with open(response_file, "a+", encoding="utf-8") as f:
                    f.write(json.dumps(save_data, ensure_ascii=False) + "\n")
                logger.info(f"成功保存请求和响应数据到: {response_file}")
        else:
            logger.warning(f'模型返回结果为空 response_data = {response_data}')
    except Exception as e:
        logger.error(f"保存请求/响应数据失败: {e}")

# API路由 - 转发请求（支持流式和非流式）
@app.post("/v1/{path:path}")
async def proxy_request(path: str, request: Request, background_tasks: BackgroundTasks):
    if not config:
        raise HTTPException(status_code=500, detail="网关配置未加载")

    request_id = f"{int(time.time())}_{id(request)}"
    request_data = await request.json()
    is_stream = request_data.get("stream", False)
    logger.info(f"接收到请求: {path}, stream={is_stream}, request_id={request_id}")

    if is_stream:
        return await handle_streaming_request(path, request_data, request, request_id, background_tasks)
    else:
        return await handle_normal_request(path, request_data, request, request_id, background_tasks)

# 处理非流式请求
async def handle_normal_request(path: str, request_data: dict, original_request: Request, request_id: str, background_tasks: BackgroundTasks):
    errors = []
    primary_instance = None
    fallback_responses = []

    # 尝试主实例
    try:
        primary_instance = select_instance(config.instances)
        instance_id = f"{primary_instance.url}_{primary_instance.model_name}"
        active_connections[instance_id] += 1
        headers = {"Content-Type": "application/json"}
        if primary_instance.api_key:
            headers["Authorization"] = f"Bearer {primary_instance.api_key}"
        auth_header = original_request.headers.get("Authorization")
        if auth_header and not primary_instance.api_key:
            headers["Authorization"] = auth_header
        async with aiohttp.ClientSession() as session:
            target_url = f"{primary_instance.url}/{path}"
            async with session.post(target_url, json=request_data, headers=headers, timeout=TIMEOUT, proxy=http_proxy) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status, detail=f"主实例 {instance_id} 返回错误: {error_text}")
                response_data = await response.json()
                background_tasks.add_task(save_request_response, request_id, primary_instance.model_name, request_data, response_data)
                return response_data
    except HTTPException as http_e:
        instance_id = f"{primary_instance.url}_{primary_instance.model_name}" if primary_instance else "主实例"
        errors.append(f"{instance_id} HTTP 错误: {str(http_e.detail)}")
        if primary_instance:
            record_error(instance_id, str(http_e.detail))
    except Exception as e:
        instance_id = f"{primary_instance.url}_{primary_instance.model_name}" if primary_instance else "主实例"
        errors.append(f"{instance_id} 错误: {str(e)}")
        if primary_instance:
            record_error(instance_id, str(e))
    finally:
        if primary_instance:
            instance_id = f"{primary_instance.url}_{primary_instance.model_name}"
            active_connections[instance_id] -= 1

    # 尝试所有备用实例
    for fallback_instance in config.fallback_instances:
        fallback_id = f"{fallback_instance.url}_{fallback_instance.model_name}"
        active_connections[fallback_id] += 1
        try:
            headers = {"Content-Type": "application/json"}
            if fallback_instance.api_key:
                headers["Authorization"] = f"Bearer {fallback_instance.api_key}"
            auth_header = original_request.headers.get("Authorization")
            if auth_header and not fallback_instance.api_key:
                headers["Authorization"] = auth_header
            async with aiohttp.ClientSession() as session:
                target_url = f"{fallback_instance.url}/{path}"
                async with session.post(target_url, json=request_data, headers=headers, timeout=TIMEOUT, proxy=http_proxy) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        errors.append(f"备用实例 {fallback_id} 返回错误: {error_text}")
                        record_error(fallback_id, error_text)
                        continue # 尝试下一个备用实例
                    response_data = await response.json()
                    background_tasks.add_task(save_request_response, request_id, fallback_instance.model_name, request_data, response_data)
                    logger.info(f"成功回退到实例 {fallback_id}")
                    return response_data # 成功返回，结束重试
        except HTTPException as http_fallback_e:
            errors.append(f"备用实例 {fallback_id} HTTP 错误: {str(http_fallback_e.detail)}")
            record_error(fallback_id, str(http_fallback_e.detail))
        except Exception as fallback_e:
            errors.append(f"备用实例 {fallback_id} 错误: {str(fallback_e)}")
            record_error(fallback_id, str(fallback_e))
        finally:
            active_connections[fallback_id] -= 1

    error_msg = "; ".join(errors)
    logger.error(f"所有实例都失败: {error_msg}")
    raise HTTPException(status_code=500, detail=f"所有模型实例都失败: {error_msg}")

# 处理流式请求
async def handle_streaming_request(path: str, request_data: dict, original_request: Request, request_id: str, background_tasks: BackgroundTasks):
    errors = []
    aggregated_chunks = [] # 用于收集所有 token 的 JSON chunk
    primary_instance = None

    async def stream_generator(instance: ModelInstance, is_fallback=False):
        nonlocal aggregated_chunks, errors
        instance_id = f"{instance.url}_{instance.model_name}"
        try:
            headers = {"Content-Type": "application/json"}
            if instance.api_key:
                headers["Authorization"] = f"Bearer {instance.api_key}"
            auth_header = original_request.headers.get("Authorization")
            if auth_header and not instance.api_key:
                headers["Authorization"] = auth_header

            req_data = dict(request_data)
            req_data["stream"] = True

            async with aiohttp.ClientSession() as session:
                target_url = f"{instance.url}/{path}"
                try:
                    async with session.post(target_url, json=req_data, headers=headers, timeout=TIMEOUT, proxy=http_proxy) as response:
                        if response.status >= 400:
                            error_text = await response.text()
                            errors.append(f"{'备用' if is_fallback else ''}实例 {instance_id} 返回错误: {error_text}")
                            record_error(instance_id, error_text)
                            return # Generator ends here if error
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line:
                                if line.startswith('data: '):
                                    data = line[6:]
                                    if data == '[DONE]':
                                        yield 'data: [DONE]\n\n'
                                        break
                                    else:
                                        try:
                                            chunk = json.loads(data)
                                            aggregated_chunks.append(chunk)
                                            yield f'data: {data}\n\n'
                                        except json.JSONDecodeError:
                                            logger.warning(f"无法解析JSON: {data}")
                                            continue
                except Exception as e_post:
                    print(f"**session.post 发生异常:** {traceback.format_exc()}") # 打印异常信息
                    raise e_post

        except Exception as e:
            errors.append(f"{'备用' if is_fallback else ''}实例 {instance_id} 流式输出错误: {str(e)}")
            record_error(instance_id, str(e))
        finally:
            active_connections[instance_id] -= 1

    async def combined_stream():
        nonlocal primary_instance, errors, aggregated_chunks
        # 尝试主实例
        try:
            primary_instance = select_instance(config.instances)
            instance_id = f"{primary_instance.url}_{primary_instance.model_name}"
            active_connections[instance_id] += 1
            async for chunk in stream_generator(primary_instance, False):
                yield chunk

            if aggregated_chunks: # 主实例成功返回数据
                background_tasks.add_task(save_request_response, request_id, primary_instance.model_name, request_data, aggregated_chunks)
                return # 主实例成功，直接返回

        except HTTPException as http_e:
            instance_id = f"{primary_instance.url}_{primary_instance.model_name}" if primary_instance else "主实例"
            errors.append(f"{instance_id} HTTP 错误: {str(http_e.detail)}")
            if primary_instance:
                record_error(instance_id, str(http_e.detail))
        except Exception as e:
            instance_id = f"{primary_instance.url}_{primary_instance.model_name}" if primary_instance else "主实例"
            errors.append(f"{instance_id} 错误: {str(e)}")
            if primary_instance:
                record_error(instance_id, str(e))
        finally:
            if primary_instance:
                instance_id = f"{primary_instance.url}_{primary_instance.model_name}"
                active_connections[instance_id] -= 1
        aggregated_chunks = [] # 清空，为fallback实例准备

        # 尝试所有备用实例
        attempted_instance_ids = set([instance_id])
        for fallback_instance in config.instances+config.fallback_instances:
            fallback_id = f"{fallback_instance.url}_{fallback_instance.model_name}"
            if fallback_id in attempted_instance_ids:
                continue # 避免重复尝试同一个实例
            attempted_instance_ids.add(fallback_id)
            active_connections[fallback_id] += 1
            aggregated_chunks = [] # Reset for each fallback instance
            try:
                async for chunk in stream_generator(fallback_instance, True):
                    yield chunk
                if aggregated_chunks: # 备用实例成功返回数据
                    logger.info(f"成功回退到实例 {fallback_id}")
                    background_tasks.add_task(save_request_response, request_id, fallback_instance.model_name, request_data, aggregated_chunks)
                    return # 备用实例成功，直接返回
            except HTTPException as http_fallback_e:
                errors.append(f"备用实例 {fallback_id} HTTP 错误: {str(http_fallback_e.detail)}")
                record_error(fallback_id, str(http_fallback_e.detail))
            except Exception as fallback_e:
                errors.append(f"备用实例 {fallback_id} 错误: {str(fallback_e)}")
                record_error(fallback_id, str(fallback_e))
            finally:
                active_connections[fallback_id] -= 1
            aggregated_chunks = [] # Reset for next fallback instance

        yield 'data: [DONE]\n\n' # 所有实例都失败，返回 [DONE] 结束流

        if errors: # Log combined errors if any
            error_msg = "; ".join(errors)
            logger.error(f"所有实例都失败: {error_msg}")


    response = StreamingResponse(combined_stream(), media_type="text/event-stream")
    return response

# 健康检查端点
@app.get("/health")
async def health_check():
    if not config:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "reason": "配置未加载"})
    if not config.instances and not config.fallback_instances:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "reason": "没有配置模型实例"})

    instance_statuses = {}
    all_instances = config.instances + config.fallback_instances
    for idx, instance in enumerate(all_instances):
        instance_id = f"{instance.url}_{instance.model_name}"
        errors_list = error_counters[instance_id]
        recent_errors = sum(1 for t in errors_list if time.time() - t < config.error_window)
        instance_statuses[instance_id] = {
            "url": instance.url,
            "model": instance.model_name,
            "active_connections": active_connections[instance_id],
            "recent_errors": recent_errors,
            "error_rate": f"{(recent_errors / max(1, len(errors_list))) * 100:.2f}%" if errors_list else "0%"
        }
    return {
        "status": "healthy",
        "instances": instance_statuses,
        "load_balancing_strategy": config.load_balancing_strategy,
        "uptime": time.time() - start_time
    }


# 添加的 models 接口
@app.get("/v1/models")
async def list_models():
    """
    Returns a list of available models.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-r1", # 示例模型ID，需要根据实际情况修改
                "object": "model",
                "created": 1741009781, # 示例时间戳
                "owned_by": "vllm", # 示例拥有者
                "root": "/data/hf_models/DeepSeek-R1-AWQ", # 示例路径
                "parent": None,
                "max_model_len": 16384, # 示例最大长度
                "permission": [
                    {
                        "id": "modelperm-2fe3878ac4644df5937f5b77678e6a85", # 示例权限ID
                        "object": "model_permission",
                        "created": 1741009781, # 示例时间戳
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ]
            }
        ]
    }


# 启动事件
@app.on_event("startup")
async def startup_event():
    global start_time, config
    start_time = time.time()
    if not load_config():
        logger.warning("启动时加载配置失败，将使用默认空配置")
        config = GatewayConfig(instances=[], fallback_instances=[]) # 初始化 fallback_instances
    event_handler = ConfigFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(os.path.abspath(config_file_path)) or ".", recursive=False)
    observer.start()
    logger.info("模型网关已启动")

if __name__ == "__main__":
    uvicorn.run("model_gateway:app", host="0.0.0.0", port=9999, reload=False)