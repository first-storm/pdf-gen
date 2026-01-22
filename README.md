# gemini-pdf-agent

一个基于 Google Gemini API 的排版代理：输入需求文本，自动生成可打印的中文 HTML/CSS，渲染为 PDF，并将每页渲染为 PNG 回馈给模型做排版自检与修订，循环多轮，最终输出 PDF。可选进行视觉回归对比并生成差异报告。

## 安装

```bash
pip install -e .
# 或
pip install .
```

### Playwright 安装 Chromium

```bash
python -m playwright install chromium
```

### 设置 GEMINI_API_KEY / OPENAI_API_KEY

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
# OpenAI 兼容端点可使用 OPENAI_API_KEY（或沿用 GEMINI_API_KEY）
# export OPENAI_API_KEY="YOUR_API_KEY"
# Windows PowerShell
# setx GEMINI_API_KEY "YOUR_API_KEY"
```

## 常用命令

```bash
# 生成 PDF（默认 2 轮）
gemini-pdf-agent --prompt examples/prompt.txt --out result.pdf

# 初始化 baseline（使用首次渲染 pages_00 作为基线）
gemini-pdf-agent --prompt examples/prompt.txt --out result.pdf --baseline baseline_pages --init-baseline

# 回归对比（通过则提前结束）
gemini-pdf-agent --prompt examples/prompt.txt --out result.pdf --baseline baseline_pages --diff-threshold 0.005
```

## HTTP 服务

启动服务：

```bash
gemini-pdf-agent-server --host 0.0.0.0 --port 8000
# 指定配置文件
gemini-pdf-agent-server --config /path/to/config.json
```

### API 文档

#### `POST /v1/render`（SSE 流式）

请求体支持两种方式：

1) JSON

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
  "storage": {
    "provider": "s3",
    "bucket": "your-bucket",
    "region": "ap-northeast-1",
    "endpoint_url": "https://s3.ap-northeast-1.amazonaws.com",
    "prefix": "pdf-jobs"
  },
  "return_pdf": "url"
}
```

2) `multipart/form-data`（上传字体/图片）

`payload` 字段为 JSON，`fonts` 为字体文件（可多个），可选 `font_families` 以指定字体族名。
如需图片，添加 `images`（可多个），可选 `image_names` 以指定图片名称（用于提示模型）。

```bash
curl -N \
  -F 'payload={"prompt":"请生成两页公司简报","model":"gemini-2.0-flash","api_key":"YOUR_KEY","use_fontconfig":true,"font_families":["MySerif"]}' \
  -F "fonts=@/path/to/MySerif.otf" \
  http://localhost:8000/v1/render

上传图片示例：

```bash
curl -N \
  -F 'payload={"prompt":"用提供的图片生成一页海报","model":"gemini-2.0-flash","api_key":"YOUR_KEY","image_names":["cover"]}' \
  -F "images=@/path/to/cover.png" \
  http://localhost:8000/v1/render
```
```

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
- `use_fontconfig` 默认 `true`，会将系统字体族名加入可用列表；客户端可显式传 `false` 关闭。
- `ttl_seconds` 默认 600，超时后会删除本地 state 和 S3/R2 对象。

`return_pdf` 可选：
- `url`（默认）：返回下载地址 `/v1/results/{job_id}`
- `base64`：在 `done` 事件里返回 `pdf_base64`

#### `POST /v1/continue`（SSE 流式）

用于在 `done` 之后追加需求并继续迭代。请求体至少需要 `job_id`，可追加 `prompt_append`。
同样支持 `multipart/form-data` 上传字体/图片（字段与 `/v1/render` 一致）。

```json
{
  "job_id": "abc123",
  "prompt_append": "把封面标题改为深蓝色，并增加目录页",
  "iterations": 2
}
```

SSE 事件与 `/v1/render` 一致，`start` 事件会额外包含 `resume_from`。

注意：`/v1/continue` 会再次调用模型，因此需要提供 `api_key`（或提前设置 `OPENAI_API_KEY` / `GEMINI_API_KEY`）。

#### `GET /v1/results/{job_id}`

下载最终 PDF（若本地已清理会从 S3/R2 读取）。

#### `POST /v1/stop`

请求停止正在进行的任务（会在当前步骤结束后停止）。可选立即清理中间文件。

```json
{
  "job_id": "abc123",
  "cleanup": true
}
```

#### `POST /v1/cleanup`

清理指定任务的本地/存储资源（如果存在）。

```json
{
  "job_id": "abc123"
}
```

#### `GET /v1/fonts`

返回系统 `fontconfig` 字体族名列表（若系统缺少 `fc-list` 会为空）。

### S3 / R2 存储

服务在渲染完成后会上传 PDF 到对象存储并清理本地文件（图片上传仅保留本地，不上传到对象存储），state 与文本内容会在 `ttl_seconds` 后删除，并同步删除对象存储内容。
继续渲染依赖本地保存的页面图；用户上传图片会从本地持久缓存目录读取。

R2 示例（S3 兼容）：

```json
{
  "storage": {
    "provider": "s3",
    "bucket": "your-r2-bucket",
    "endpoint_url": "https://<accountid>.r2.cloudflarestorage.com",
    "region": "auto",
    "prefix": "pdf-jobs"
  }
}
```

凭据优先使用 `storage.access_key_id/secret_access_key`，否则读取环境变量：
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`。

## 交互模式与配置

- 交互输入 prompt：

```bash
gemini-pdf-agent -i --out result.pdf
# 输入文本后按 Ctrl-D (Windows: Ctrl-Z) 结束
```

- 直接在命令行输入 prompt：

```bash
gemini-pdf-agent --prompt-text "请生成两页公司简报" --out result.pdf
```

- 保存默认配置（模型、base url、key、迭代次数等）到 `~/.config/gemini-pdf-agent/config.json`：

```bash
gemini-pdf-agent --save-config --model gemini-2.0-flash --base-url https://your-endpoint --iterations 2 --api-key YOUR_KEY
```

### 字体白名单（配置文件）

可在 `~/.config/gemini-pdf-agent/config.json` 配置允许使用的字体列表，模型只允许输出这些字体。

示例：

```json
{
  "api_mode": "openai",
  "temperature": 1.0,
  "reasoning_effort": "medium",
  "allowed_fonts": ["Noto Serif CJK SC", "Source Han Serif SC"],
  "font_files": ["MySerif::/path/to/MySerif.otf"],
  "use_fontconfig": true,
  "max_image_bytes": 5000000,
  "max_image_count": 10
}
```

- `allowed_fonts`: 允许模型使用的字体族名。
- `font_files`: 指定字体文件，格式为 `字体族名::路径`（族名可省略，省略时默认使用文件名作为族名）。
- `use_fontconfig`: 读取系统 fontconfig 字体族名（若系统没有 `fc-list` 会自动忽略）。
- `api_mode`: 选择 `gemini` 或 `openai`（OpenAI 兼容端点）。
- `temperature`: OpenAI 兼容端点的采样温度。
- `reasoning_effort`: OpenAI 兼容端点的推理强度（如 `low` / `medium` / `high`）。
- `max_image_bytes`: 单张图片大小上限（字节），默认 `5000000`，设置为 `0` 可关闭限制。
- `max_image_count`: 单次任务图片数量上限，默认 `10`，设置为 `0` 可关闭限制。

## 字体说明（中文 serif）

- 下载思源宋体或 Noto Serif CJK 字体文件，并通过 `--cjk-font /path/to/font.otf` 传入。
- 常见坑：路径含空格需要加引号；Windows 请使用完整路径（如 `C:\\Fonts\\SourceHanSerif.otf`）。

## 产物目录（workdir）结构

```
run_YYYYMMDD_HHMMSS/
  draft_00.html
  draft_00.pdf
  pages_00/
    page_001.png
  draft_01.html
  draft_01.pdf
  pages_01/
  diff/
    diff_page_001.png
  regression_report.json
```

## 说明

- 默认渲染后端为 Playwright Chromium；如需 WeasyPrint：
- 模型在迭代评审阶段可返回 `done: true` 提前结束迭代并复用上一次输出。
- 当模型请求结束时，程序会询问是否结束；可输入额外提示继续迭代。

```bash
pip install .[weasyprint]
```

WeasyPrint 可能需要系统依赖，请参考其官方文档。

## 参数示例

```bash
gemini-pdf-agent \
  --prompt examples/prompt.txt \
  --out result.pdf \
  --iterations 2 \
  --backend playwright \
  --model gemini-2.0-flash \
  --base-url https://your-gemini-endpoint.example.com \
  --cjk-font /path/to/font.otf \
  --baseline baseline_pages \
  --diff-threshold 0.005
```

## OpenAI 兼容 /v1/chat/completions

- 通过 `--api-mode openai` 启用 OpenAI 兼容端点。
- `--base-url` 可直接填完整的 `/v1/chat/completions`，或填 API 根地址（会自动补全）。

示例：

```bash
gemini-pdf-agent \
  --prompt examples/prompt.txt \
  --out result.pdf \
  --api-mode openai \
  --model gpt-4o-mini \
  --base-url https://api.openai.com/v1 \
  --api-key YOUR_KEY
```
