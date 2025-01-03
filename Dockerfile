# 使用Python 3.10作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 创建必要的目录并设置权限
RUN mkdir -p images cache && \
    chmod 777 images cache

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps git+https://github.com/apple/ml-mobileclip.git

# 复制应用代码和资源
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "120", "--workers", "1"] 