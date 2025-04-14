FROM python:3.12-slim

WORKDIR /app

# 安装必要的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 添加应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令（开发模式，支持热重载，设置2秒延迟）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-delay", "15"]