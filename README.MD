# Gemini 2.5 PlayGround 
# Gemini 2.5 游乐场

# [English Version](README_EN.MD)

### 作者：技术爬爬虾
B站，YouTube，抖音，公众号 全网同名。转载请注明作者。

## 项目简介
#### Demo: [https://play.210718.xyz](https://play.210718.xyz)
只需准备一个Gemini API Key，10秒即可部署一个Gemini多模态对话的网站。
不限地区/网络环境，打开即用，<b>适配了手机端</b>。
使用Deno/Cloudflare Worker无服务器免费部署。
对Gemini Chat API 进行了格式转换，中转为更通用的OpenAI格式，国内直连可用。
可对接AI编程，接入ChatBox等AI客户端。

## Deno部署（推荐）

0. 准备一个Gemini API Key [https://aistudio.google.com](https://aistudio.google.com)（免费）
1. [fork](https://github.com/tech-shrimp/gemini-playground/fork)本项目
2. 登录/注册 https://dash.deno.com/
3. 创建项目 https://dash.deno.com/new_project
4. 选择此项目，填写项目名字（请仔细填写项目名字，关系到自动分配的域名）
5. Entrypoint 填写 `src/deno_index.ts` 其他字段留空 
   <details>
   <summary>如图</summary>
   
   ![image](/docs/images/1.png)
   </details>
6. 点击 <b>Deploy Project</b>
7. 部署成功后获得域名，点开即用。域名同样可以作为Chat API的代理使用。

## Cloudflare Worker 部署

[![Deploy to Cloudflare Workers](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/tech-shrimp/gemini-playground)

0. 准备一个Gemini API Key [https://aistudio.google.com](https://aistudio.google.com)（免费）
1. 点击部署按钮
2. 登录Cloudflare账号
3. 填入Account ID，与API Token
4. Fork本项目，开启Github Action功能
5. 部署，打开dash.cloudflare.com，查看部署后的worker
6. 国内使用需要绑定一个自定义域名
   <details>
   <summary>如图</summary>
   
   ![image](/docs/images/3.png)
   </details>
> 国内使用cloudflare有时候可能出现400: User location is not supported for the API use. 可能是粤港地区Cloudflare路由到了香港的CDN节点代理导致的。建议换成Deno部署。

## 本地调试

Windows 安装Deno:
> irm https://deno.land/install.ps1 | iex

Mac/Linux 安装Deno:
> curl -fsSL https://deno.land/install.sh | sh

启动项目：

>cd 项目目录 <br>
>deno run --allow-net --allow-read src/deno_index.ts


## 使用方法
### 多模态对话
填入API Key, 点击Connect按钮即可对话。
以下三个按钮分别是：
- 启用麦克风
- 启用摄像头
- 分享屏幕
<br>

<details>
   <summary>如图</summary>

   ![image](/docs/images/2.png)
</details>


### API 代理

#### 双入口模式

项目提供两个独立入口，互不干扰：

| 入口 | 路径 | 说明 |
|---|---|---|
| **OpenAI 兼容** | `/openai/v1/chat/completions` | OpenAI 格式请求自动转换为 Gemini 格式，适合 ChatBox / Cursor 等 OpenAI 客户端 |
| **Gemini 原生** | `/gemini/v1beta/...` | 原生格式**零转换透传**，thought_signature 等由客户端 SDK 管理，适合支持 Gemini 原生接口的客户端（也兼容 `/gemni` 拼写） |

无前缀路径（`/v1/chat/completions`、`/v1beta/...`）自动识别，向后兼容。

API已经被代理为OpenAI格式，可以直接使用OpenAI格式的API。
不限网络环境，可免梯子使用。
注意替换域名与Gemini API Key。

#### 可用模型列表：
```
curl --location 'http://your.domain.com/v1/models' \
--header 'Authorization: Bearer YOUR-GEMINI-API-KEY'
```

#### 对话:
```
curl --location 'https://your.domain.com/v1/chat/completions' \
--header 'Authorization: Bearer YOUR-GEMINI-API-KEY' \
--header 'Content-Type: application/json' \
--data '{
    "messages": [
        {
            "role": "system",
            "content": "You are a test assistant."
        },
        {
            "role": "user",
            "content": "Testing. Just say hi and nothing else."
        }
    ],
    "model": "gemini-3-flash"
}'
```

#### 函数调用（Function Calling）支持：

代理已完整支持 OpenAI 格式的 Function Calling（工具调用），可对接 MCP、Shell 等工具：

- `tools` → Gemini `functionDeclarations`（支持并行函数调用）
- `tool_choice` → Gemini `toolConfig`（`auto` / `none` / `required` / 指定函数）
- `role: "tool"` 消息 → Gemini `functionResponse`（多轮工具调用链路完整）
- 流式（SSE）下同样支持 `tool_calls` 增量输出
- 兼容旧版 `functions` 参数
- 可选 `thinking` 配置透传（Gemini 3 原生思考控制，如 `"thinking": {"thinkingBudget": 0}`）

> 注意：`gemini-2.x` / `gemini-1.5` 系列模型已退役（API 返回 404），请使用 `gemini-3-flash` / `gemini-3.1-flash-lite` 等 3.x 模型。代理默认模型为 `gemini-3-flash`。

#### 功能测试：

```bash
node test-worker.mjs
```

### ChatBOX：
   <details>
   <summary>如图</summary>
   
   ![image](/docs/images/4.png)
   </details>

### Cursor编程：
   <details>
   <summary>如图</summary>
   
   ![image](/docs/images/5.png)
   </details>



## 致谢/引用：

网站UI： 
```
Author: ChrisKyle
Project: https://github.com/ViaAnthroposBenevolentia/gemini-2-live-api-demo
MIT License : https://github.com/ViaAnthroposBenevolentia/gemini-2-live-api-demo/blob/main/LICENSE
```

Gemini转成OpenAI格式： 
```
Author: PublicAffairs
Project: https://github.com/PublicAffairs/openai-gemini
MIT License : https://github.com/PublicAffairs/openai-gemini/blob/main/LICENSE
```
