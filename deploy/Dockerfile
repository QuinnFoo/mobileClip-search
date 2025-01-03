FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 创建必要的目录
RUN mkdir -p images cache checkpoints

# 复制应用文件
COPY requirements.txt .
COPY app.py .
COPY templates templates/
COPY checkpoints checkpoints/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps git+https://github.com/apple/ml-mobileclip.git

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 设置目录权限
RUN chmod -R 777 images cache

# 暴露端口
EXPOSE 5000

# 启动应用
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "120", "--workers", "1", "--log-level", "debug"] 