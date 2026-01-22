FROM mcr.microsoft.com/playwright:v1.57.0-noble

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk fonts-noto-cjk-extra \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["gemini-pdf-agent-server", "--host", "0.0.0.0", "--port", "8000"]
