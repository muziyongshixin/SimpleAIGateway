![本地图片](logo.jpg)

# SimpleAIGateway [English Introduction](README.md)
这是一个用Python实现的简单AI网关，具备动态配置、负载均衡、错误报警、灾难恢复和请求记录持久化等功能。

这个项目的动机是，我通过VLLM部署了多个LLM的模型实例，但是想通过一个统一的网关访问这些模型，同时实现动态配置、负载均衡、错误告警、灾难恢复和请求记录备份功能。
在进行简单的调研后，我发现大部分现有的项目都是client侧的聚合调用，很少有在server侧的提供模型网关的功能。
虽然也有一些项目能够实现类似的功能，但是往往需要进行复杂的配置和环境依赖，不太能满足我的需求，因此我决定使用python开发了这个简单的AI网关。

如果这个项目对你有帮助，欢迎star或者fork，也欢迎提交PR。

如果你发现有更好用的开源项目能够满足以上功能欢迎在issue中告诉我:)


## features
- 简单易用，这个AI网关纯python代码编写只有数百行代码，兼容OpenAI的API接口，同时支持流式和非流式输出
- 支持动态加载配置文件，新增的部署实例直接修改文件不影响线上客户端使用
- 支持负载均衡，可以通过轮询或者最少连接数等策略实现多实例的负载均衡
- 支持配置容灾兜底，可以配置外部api等方式实现容灾兜底，防止私有部署的实例都挂掉后影响线上业务
- 支持请求落表和日志记录，通过AI网关的请求和结果会自动写入到文件中，每小时保存一个文件，sharegpt格式数据，方便后续模型训练。
- 监控报警（这部分功能目前只是将错误信息写到日志文件中，可以根据自身需求加入微信发消息或者发邮件等内部的报警机制）


## Quick Start

### 环境准备：
 `pip install asyncio json5 aiohttp uvicorn fastapi pydantic watchdog openai`

 ### 启动服务：
 1. 修改配置文件config.json5，配置AI模型的地址、端口、负载均衡策略等
 <details>
 <summary>点击展开配置文件示例</summary>
 
 ```json5
 {
    "instances": [ // private instances for inference
      {
        "url": "http://10.82.1.1:8080/v1",
        "api_key": "empty",
        "model_name": "deepseek-r1",
        "weight": 1
      },
       {
       "url": "http://10.82.1.2:8080/v1",
       "api_key": "empty",
       "model_name": "deepseek-r1",
       "weight": 1
      }
    ],
    "fallback_instances":[  // optional, fallback instances for in case of all private instances are down
      {
        "url": "https://cloud.infini-ai.com/maas/v1",
        "api_key":"your_api_key",
        "model_name": "deepseek-r1",
        "weight": 1
      }

    ],
    "data_dir": "./data",
    "load_balancing_strategy": "round_robin", // options: round_robin, least_connections
    "error_threshold": 10,  // number of consecutive errors before remove this server from the pool
    "error_window": 300, // time window for error count
    "alert_cooldown": 300 // time window for alerting
  }
  ```

  </details>

 2. 启动服务：`python model_gateway.py`
 3. 访问网关服务：`http://your_model_gateway_ip:9999/`
 4. 调用示例，和标准调用VLLM或者OpenAI的API接口完全一致：
 <details>
 <summary>点击展开示例代码</summary>

 ```python
from openai import OpenAI
import time
import concurrent
import traceback

MODEL_NAME = 'deepseek-r1' 
ip_mapping = {
    'deepseek-r1':"http://your_model_gateway_ip:9999/v1",
              }

url = ip_mapping[MODEL_NAME]
client = OpenAI(
    base_url=url,
    api_key="EMPTY",
)
 
def call_one_req(messages=None, stream=False, print_process=False):
    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "1+1=？ "},
            ] if messages is None else messages,
            temperature=0.6,
            stream=stream,
            max_tokens=4096
        )

        result = ''
        if stream:
            for chunk in completion:
                if len(chunk.choices)>0:
                    reasoning_content = chunk.choices[0].delta.reasoning_content if hasattr(chunk.choices[0].delta,"reasoning_content")  else None
                    answer_content = chunk.choices[0].delta.content
                    tmp = reasoning_content if reasoning_content is not None else answer_content
                    result += tmp
                    if print_process:
                        print(tmp, end='', flush=True)
        else:
            result = completion.choices[0].message.content
            if print_process:
                print(result)

        return result
    except:
        traceback.print_exc()
        print("error")
        return None

messages = [{'role':'user','content':'hello.'}]
call_one_req(messages,stream=True, print_process=True)
```
</details>


5. 访问日志文件：所有调用请求和模型返回的结果会保存在`data/requests_responses/YYYYMMDD_HH.jsonl` 文件中，每小时保存一个文件，sharegpt格式数据，方便后续模型训练。

## Risks
1. 该项目目前只确认支持OpenAI的API接口，其他的接口没有经过测试，可能会有兼容性问题。
2. 代码目前没有在大规模高流量场景下测试，不确定是否存在性能问题。