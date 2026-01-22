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
  "use_fontconfig": true
}
```

- `allowed_fonts`: 允许模型使用的字体族名。
- `font_files`: 指定字体文件，格式为 `字体族名::路径`（族名可省略，省略时默认使用文件名作为族名）。
- `use_fontconfig`: 读取系统 fontconfig 字体族名（若系统没有 `fc-list` 会自动忽略）。
- `api_mode`: 选择 `gemini` 或 `openai`（OpenAI 兼容端点）。
- `temperature`: OpenAI 兼容端点的采样温度。
- `reasoning_effort`: OpenAI 兼容端点的推理强度（如 `low` / `medium` / `high`）。

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
