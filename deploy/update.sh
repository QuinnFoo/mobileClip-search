#!/bin/bash

# 检查必要的环境变量
if [ -z "$EC2_IP" ] || [ -z "$EC2_KEY" ]; then
    echo "Error: EC2_IP or EC2_KEY environment variable is not set"
    exit 1
fi

# 上传更新的文件
echo "Uploading updated files..."
scp -i "$EC2_KEY" ../requirements.txt "ubuntu@${EC2_IP}:/home/ubuntu/app/"
scp -i "$EC2_KEY" Dockerfile "ubuntu@${EC2_IP}:/home/ubuntu/app/"

# 重新构建和启动服务
echo "Rebuilding and restarting services..."
ssh -i "$EC2_KEY" "ubuntu@${EC2_IP}" "cd /home/ubuntu/app && sudo docker-compose down && sudo docker-compose up --build -d"

echo "Update completed!" 