# llm-mcp: local-model MCP tools (see server.py and docs/clients.md).
FROM python:3.12-slim
WORKDIR /app
# libmagic: inspect_file type detection; ffmpeg (ffprobe): audio/video structure
RUN apt-get update && apt-get install -y --no-install-recommends libmagic1 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
RUN useradd -u 1000 -m app
USER app
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["python", "server.py"]
