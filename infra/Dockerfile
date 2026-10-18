ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUTF8=1 \
    WEB_HOST=0.0.0.0 \
    API_HOST=0.0.0.0 \
    API_PORT=8000
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
# CPU-only fixed version. No torchvision or torchaudio used by this app.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
    && pip install -r /app/requirements.txt \
    && pip check
COPY . /app
EXPOSE 8501 8000
CMD ["python", "-m", "utils.launch", "web"]
