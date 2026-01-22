FROM mcr.microsoft.com/playwright/python:v1.45.0-jammy

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["gemini-pdf-agent-server", "--host", "0.0.0.0", "--port", "8000"]
