#!/bin/bash

# 加载环境变量
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found!"
    exit 1
fi

# 转换 Windows 路径为 MSYS 路径
if [[ "$OSTYPE" == "msys" ]]; then
    echo "Converting Windows paths for MSYS..."
    # 移除引号并转换反斜杠
    EC2_KEY=$(echo $EC2_KEY | tr -d '"' | sed 's/\\/\//g')
    # 转换驱动器号格式
    EC2_KEY=$(echo $EC2_KEY | sed 's/^\([A-Za-z]\):/\/\1/')
    echo "Converted EC2_KEY path: $EC2_KEY"
fi

# 检查 SSH 密钥文件
if [ ! -f "$EC2_KEY" ]; then
    echo "Error: SSH key file not found at: $EC2_KEY"
    echo "Please make sure the key file exists and has correct permissions"
    exit 1
fi

# 检查 SSH 密钥文件权限
if [[ "$OSTYPE" != "msys" ]]; then
    if [ "$(stat -c %a "$EC2_KEY")" != "400" ]; then
        echo "Fixing SSH key file permissions..."
        chmod 400 "$EC2_KEY"
    fi
fi

# 检查必要的环境变量
required_vars=(
    "AWS_ACCESS_KEY_ID"
    "AWS_SECRET_ACCESS_KEY"
    "AWS_DEFAULT_REGION"
    "S3_BUCKET"
    "EC2_IP"
    "EC2_KEY"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not set"
        exit 1
    fi
done

echo "Environment variables loaded successfully!"

# 测试 SSH 连接
echo "Testing SSH connection..."
if ! ssh -i "$EC2_KEY" -o StrictHostKeyChecking=no "ubuntu@${EC2_IP}" "echo 'SSH connection successful'"; then
    echo "Error: Failed to connect to EC2 instance"
    echo "Please check your EC2 instance is running and security group allows SSH access"
    exit 1
fi

# 清理EC2上的空间
echo "Cleaning up EC2 instance..."
ssh -i "$EC2_KEY" "ubuntu@${EC2_IP}" "\
    echo 'Cleaning up Docker resources...' && \
    sudo docker system prune -af --volumes && \
    echo 'Cleaning up system packages...' && \
    sudo apt-get clean && \
    sudo apt-get autoremove -y && \
    echo 'Cleaning up logs...' && \
    sudo journalctl --vacuum-time=1d && \
    echo 'Checking available space...' && \
    df -h"

# 检查必要文件是否存在
required_files=(
    "app.py"
    "requirements.txt"
    "deploy/Dockerfile"
    "deploy/docker-compose.yml"
    "checkpoints/mobileclip_s0.pt"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file $file not found!"
        exit 1
    fi
done

# 准备部署文件
echo "Preparing deployment files..."
rm -rf deploy_tmp
mkdir -p deploy_tmp
cp -r app.py requirements.txt templates checkpoints deploy_tmp/
cp deploy/Dockerfile deploy_tmp/
cp deploy/docker-compose.yml deploy_tmp/

# 确保目录存在
echo "Creating necessary directories..."
ssh -i "$EC2_KEY" "ubuntu@${EC2_IP}" "mkdir -p ~/app/images ~/app/cache ~/app/checkpoints"

# 上传文件到EC2
echo "Uploading files to EC2..."
scp -i "$EC2_KEY" -r deploy_tmp/* "ubuntu@${EC2_IP}:~/app/"

# 在EC2上构建和启动应用
echo "Building and starting application on EC2..."
ssh -i "$EC2_KEY" "ubuntu@${EC2_IP}" "cd ~/app && \
    export AWS_ACCESS_KEY_ID='${AWS_ACCESS_KEY_ID}' && \
    export AWS_SECRET_ACCESS_KEY='${AWS_SECRET_ACCESS_KEY}' && \
    export AWS_DEFAULT_REGION='${AWS_DEFAULT_REGION}' && \
    export S3_BUCKET='${S3_BUCKET}' && \
    sudo -E docker-compose down && \
    sudo -E docker-compose build --no-cache && \
    sudo -E docker-compose up -d"

# 检查应用状态
echo "Checking application status..."
sleep 5
ssh -i "$EC2_KEY" "ubuntu@${EC2_IP}" "cd ~/app && sudo docker-compose ps"

# 显示应用日志
echo "Showing recent application logs..."
ssh -i "$EC2_KEY" "ubuntu@${EC2_IP}" "cd ~/app && sudo docker-compose logs --tail=100"

# 清理临时文件
echo "Cleaning up temporary files..."
rm -rf deploy_tmp

echo "Deployment completed!"
echo "You can access the application at: http://${EC2_IP}"
echo "To view logs, run: ssh -i \"${EC2_KEY}\" \"ubuntu@${EC2_IP}\" \"cd ~/app && sudo docker-compose logs -f\""