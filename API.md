# gemini-pdf-agent API

基础信息：
- 服务 Base URL: `http://<host>:<port>`（本项目 HTTP 服务地址）
- 响应格式：JSON 或 SSE（`text/event-stream`）
- 认证：通过请求体中的 `api_key`，或提前设置环境变量 `GEMINI_API_KEY` / `OPENAI_API_KEY`
注意：
- `base_url` 是模型 API 的兼容端点地址（例如代理/自建网关），不是本服务的地址。

## `GET /v1/health`

健康检查。

响应示例：

```json
{"status":"ok"}
```

## `POST /v1/render`（SSE 流式）

发起渲染任务并实时返回事件流。支持 JSON 与 `multipart/form-data`。

请求字段（JSON payload）：

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| prompt | string | 是 | - | 文本需求，不能为空 |
| model | string | 否 | `gemini-2.0-flash` | 模型名称 |
| base_url | string | 否 | - | 兼容端点地址 |
| api_mode | string | 否 | - | API 模式（如 `gemini`） |
| api_key | string | 否 | - | API Key（不提供则读取环境变量） |
| iterations | number | 否 | `2` | 迭代次数，最小为 1 |
| backend | string | 否 | `playwright` | 渲染后端 |
| zoom | number | 否 | `2.0` | PNG 放大倍数 |
| temperature | number | 否 | - | 模型采样温度 |
| reasoning_effort | string | 否 | - | 推理力度（如 `low/medium/high`） |
| allowed_fonts | string[] | 否 | `[]` | 允许的字体族名 |
| font_files | string[] | 否 | `[]` | 已存在字体文件路径（服务端路径） |
| use_fontconfig | boolean | 否 | `true` | 是否读取系统字体族名 |
| cjk_font | string | 否 | - | CJK 默认字体族名 |
| ttl_seconds | number | 否 | `600` | 任务与 state 的保留时长 |
| return_pdf | string | 否 | `url` | `url` 或 `base64` |
| workdir | string | 否 | - | 自定义工作目录 |
| job_id | string | 否 | 自动生成 | 自定义 job id |
| max_image_bytes | number | 否 | `5000000` | 单张图片最大字节数，<=0 表示不限制 |
| max_image_count | number | 否 | `10` | 允许图片数量，<=0 表示不限制 |
| storage | object | 否 | - | 仅服务端配置生效，客户端提供会被忽略 |

JSON 示例：

```json
{
  "prompt": "请生成两页公司简报",
  "model": "gemini-2.0-flash",
  "base_url": "https://your-endpoint",
  "api_mode": "gemini",
  "api_key": "YOUR_KEY",
  "iterations": 2,
  "backend": "playwright",
  "zoom": 2.0,
  "temperature": 1.0,
  "reasoning_effort": "medium",
  "allowed_fonts": ["Noto Serif CJK SC"],
  "font_files": [],
  "use_fontconfig": true,
  "ttl_seconds": 600,
  "return_pdf": "url",
  "max_image_bytes": 5000000,
  "max_image_count": 10
}
```

`multipart/form-data`：

- `payload`：JSON 字符串
- `fonts`：字体文件（可多个）
- `font_families`：可选，字体族名数组，与 `fonts` 一一对应
- `images`：图片文件（可多个）
- `image_names`：可选，图片名称数组，用于提示模型

上传字体示例：

```bash
curl -N \
  -F 'payload={"prompt":"请生成两页公司简报","model":"gemini-2.0-flash","api_key":"YOUR_KEY","use_fontconfig":true,"font_families":["MySerif"]}' \
  -F "fonts=@/path/to/MySerif.otf" \
  http://localhost:8000/v1/render
```

上传图片示例：

```bash
curl -N \
  -F 'payload={"prompt":"用提供的图片生成一页海报","model":"gemini-2.0-flash","api_key":"YOUR_KEY","image_names":["cover"]}' \
  -F "images=@/path/to/cover.png" \
  http://localhost:8000/v1/render
```

SSE 事件：

| 事件 | data 字段 | 说明 |
| --- | --- | --- |
| start | job_id, workdir | 任务开始 |
| iteration_start | iteration, iterations | 迭代开始 |
| rendered | iteration, pdf_path, page_count | 单轮渲染完成 |
| issues | iteration, issues | 模型发现的问题列表 |
| changes | iteration, changes | 模型改动说明 |
| early_stop | iteration, reason | 提前停止（如 `model_done`） |
| storage | status, bucket, result_key, result_url | 上传对象存储状态 |
| done | job_id, pdf_path, download_url, storage?, pdf_base64? | 完成 |
| error | message | 失败 |
| stopped | job_id, cleanup | 被停止 |

SSE 事件示例：

```
event: start
data: {"job_id":"abc123","workdir":"server_runs/run_..."}

event: iteration_start
data: {"iteration":1,"iterations":2}

event: rendered
data: {"iteration":1,"pdf_path":"...","page_count":2}

event: done
data: {"job_id":"abc123","pdf_path":".../result.pdf","download_url":"/v1/results/abc123"}
```

说明：
- `use_fontconfig` 为 `true` 时会自动并入系统字体族名。
- `return_pdf=base64` 时，`done` 事件包含 `pdf_base64`。
- 上传的图片会被保留在本地缓存目录；任务完成后仍可用于 `continue`。

## `POST /v1/continue`（SSE 流式）

用于在 `done` 之后追加需求并继续迭代。至少提供 `job_id`。
支持 `multipart/form-data`，字段与 `/v1/render` 一致。

请求字段（额外部分）：

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| job_id | string | 是 | - | 续写的任务 id |
| prompt_append | string | 否 | - | 追加需求（推荐） |
| iterations | number | 否 | `2` | 追加迭代次数 |
| resume_from | string | 否 | - | 标注来源 job id（服务端会回传） |

JSON 示例：

```json
{
  "job_id": "abc123",
  "prompt_append": "把封面标题改为深蓝色，并增加目录页",
  "iterations": 2
}
```

SSE 事件与 `/v1/render` 一致，`start` 事件会额外包含 `resume_from`。

注意：`/v1/continue` 会再次调用模型，因此需要提供 `api_key`（或提前设置 `OPENAI_API_KEY` / `GEMINI_API_KEY`）。

## `GET /v1/results/{job_id}`

下载最终 PDF。若本地已清理，会从对象存储读取（若启用）。

响应：`application/pdf`

## `POST /v1/stop`

请求停止正在进行的任务（会在当前步骤结束后停止）。可选立即清理中间文件。

请求示例：

```json
{
  "job_id": "abc123",
  "cleanup": true
}
```

响应示例：

```json
{"status":"stopping","job_id":"abc123","cleanup":true}
```

## `POST /v1/cleanup`

清理指定任务的本地/存储资源（如果存在）。

请求示例：

```json
{
  "job_id": "abc123"
}
```

响应示例：

```json
{"status":"cleaned","job_id":"abc123"}
```

## `GET /v1/fonts`

返回系统 `fontconfig` 字体族名列表（若系统缺少 `fc-list` 会为空）。

响应示例：

```json
{"fontconfig":["Noto Sans CJK SC","Source Han Serif SC"]}
```

## 错误响应约定

HTTP 400/404：

```json
{"detail":"error message"}
```

SSE 运行时错误：

```
event: error
data: {"message":"error message"}
```
