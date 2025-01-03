import os
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
import torch
from PIL import Image
import mobileclip
import numpy as np
from pathlib import Path
import faiss
import io
import base64
import pickle
from datetime import datetime

app = Flask(__name__)

# 存储配置
STORAGE_TYPE = os.getenv('STORAGE_TYPE', 'local')  # 默认使用本地存储，可选 'local' 或 's3'
IMAGE_LIBRARY_DIR = 'images'  # 本地图片目录
os.makedirs(IMAGE_LIBRARY_DIR, exist_ok=True)

# S3 配置（仅在使用 S3 时初始化）
if STORAGE_TYPE == 's3':
    import boto3
    from botocore.exceptions import ClientError
    
    # 获取 AWS 配置
    S3_BUCKET = os.getenv('S3_BUCKET', 'qf-clip-images')
    AWS_REGION = os.getenv('AWS_REGION', 'ap-southeast-2')  # 默认使用悉尼区域
    
    try:
        print(f"Initializing S3 client with region: {AWS_REGION}")
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        # 测试连接
        s3_client.list_buckets()
        print("Successfully connected to S3")
    except Exception as e:
        print(f"Error initializing S3 client: {e}")
        print("Falling back to local storage")
        STORAGE_TYPE = 'local'

# 加载模型
model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='checkpoints/mobileclip_s0.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')
model.eval()

class ImageLibrary:
    def __init__(self):
        self.image_paths = []
        self.image_features = []
        self.index = None
        self.feature_cache_file = 'cache/feature_vectors.pkl'
        self.image_hash_file = 'cache/image_hashes.pkl'
        self.processed_files = {}  # 使用字典存储文件名到索引的映射
        self.load_cache()
    
    def load_cache(self):
        """加载缓存的特征向量"""
        if os.path.exists(self.feature_cache_file):
            try:
                with open(self.feature_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.image_paths = cache_data.get('paths', [])
                    self.image_features = cache_data.get('features', [])
                    # 重建处理文件的映射
                    self.processed_files = {
                        os.path.basename(path): idx 
                        for idx, path in enumerate(self.image_paths)
                    }
                    print(f"Loaded {len(self.image_paths)} cached features")
                    if self.image_features:
                        self.build_index()
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.image_paths = []
                self.image_features = []
                self.processed_files = {}
    
    def save_cache(self):
        """保存特征向量到缓存"""
        try:
            os.makedirs('cache', exist_ok=True)
            cache_data = {
                'paths': self.image_paths,
                'features': self.image_features
            }
            with open(self.feature_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved {len(self.image_paths)} features to cache")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def add_image(self, image_path, feature_vector):
        """添加新图片到库中，如果已存在则跳过"""
        filename = os.path.basename(image_path)
        
        if filename in self.processed_files:
            # 如果文件已存在，直接跳过
            print(f"Skipping existing file: {filename}")
            return False
        
        # 添加新文件
        idx = len(self.image_paths)
        self.image_paths.append(image_path)
        self.image_features.append(feature_vector)
        self.processed_files[filename] = idx
        print(f"Added new image: {filename}")
        return True
    
    def build_index(self):
        """构建FAISS索引"""
        if not self.image_features:
            print("No features to build index")
            return
        try:
            features = np.array(self.image_features)
            d = features.shape[1]  # 向量维度
            self.index = faiss.IndexFlatIP(d)  # 使用内积相似度
            self.index.add(features)
            print(f"Built index with {len(self.image_features)} vectors")
        except Exception as e:
            print(f"Error building index: {e}")
            self.index = None
    
    def search(self, query_vector, k=3, threshold=0.2):
        """搜索相似图片"""
        if not self.image_features:
            print("No images in library")
            return []
            
        if self.index is None:
            print("Building index for search")
            self.build_index()
            if self.index is None:
                return []
        
        try:
            scores, indices = self.index.search(query_vector.reshape(1, -1), k)
            
            # 只返回相似度大于阈值的结果
            filtered_results = [
                (self.image_paths[idx], score)
                for idx, score in zip(indices[0], scores[0])
                if score >= threshold
            ]
            
            print(f"Found {len(filtered_results)} results above threshold {threshold}")
            return filtered_results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def get_image_hash(self, img_path):
        """获取图片的修改时间和大小作为简单的哈希"""
        stat = os.stat(img_path)
        return f"{stat.st_mtime}_{stat.st_size}"

    def load_image_hashes(self):
        """加载已处理图片的哈希值"""
        if os.path.exists(self.image_hash_file):
            with open(self.image_hash_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_image_hashes(self, hashes):
        """保存图片哈希值"""
        with open(self.image_hash_file, 'wb') as f:
            pickle.dump(hashes, f)

# 创建图片库实例
image_library = ImageLibrary()

def process_image(image):
    """处理图片并返回特征向量"""
    img_tensor = preprocess(image.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]

def process_text(text):
    """处理文本并返回特征向量"""
    text_token = tokenizer([text])
    with torch.no_grad():
        features = model.encode_text(text_token)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0]

def upload_to_s3(file_path, bucket, object_name=None):
    """上传文件到 S3（仅在 S3 模式下使用）"""
    if STORAGE_TYPE != 's3':
        return True
    
    if object_name is None:
        object_name = os.path.basename(file_path)

    try:
        s3_client.upload_file(file_path, bucket, object_name)
        return True
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return False

def download_from_s3(bucket, object_name, file_path):
    """从 S3 下载文件（仅在 S3 模式下使用）"""
    if STORAGE_TYPE != 's3':
        return True
    
    try:
        s3_client.download_file(bucket, object_name, file_path)
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return False

def get_s3_presigned_url(object_name, expiration=3600):
    """获取 S3 对象的预签名 URL"""
    try:
        url = s3_client.generate_presigned_url('get_object',
                                            Params={'Bucket': S3_BUCKET,
                                                    'Key': object_name},
                                            ExpiresIn=expiration)
        return url
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None

def build_image_library():
    """构建图片库（支持本地和 S3 模式）"""
    print("\n=== Starting Image Library Build ===")
    new_images_added = False
    current_hashes = {}

    if STORAGE_TYPE == 's3':
        try:
            print("\nRunning in S3 mode, scanning bucket:", S3_BUCKET)
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=S3_BUCKET):
                for obj in page.get('Contents', []):
                    filename = obj['Key']
                    if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                        # 检查文件是否已处理
                        if filename in image_library.processed_files:
                            print(f"Skipping existing file: {filename}")
                            continue
                        
                        try:
                            # 直接从 S3 读取图片数据
                            print(f"Processing new image: {filename}")
                            response = s3_client.get_object(Bucket=S3_BUCKET, Key=filename)
                            image_data = response['Body'].read()
                            image = Image.open(io.BytesIO(image_data))
                            
                            # 处理图片并添加到库中
                            features = process_image(image)
                            if image_library.add_image(filename, features):  # 直接使用 S3 key 作为路径
                                new_images_added = True
                            
                            # 使用 S3 对象的元数据作为哈希
                            current_hashes[filename] = f"{obj['LastModified']}_{obj['Size']}"
                            
                        except Exception as e:
                            print(f"Error processing {filename}: {e}")
                            
        except Exception as e:
            print(f"Error accessing S3: {e}")
    else:
        # 本地模式
        print(f"\nScanning directory: {IMAGE_LIBRARY_DIR}")
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in Path(IMAGE_LIBRARY_DIR).glob(ext):
                filename = os.path.basename(img_path)
                img_path_str = str(img_path)
                
                # 检查文件是否已处理
                if filename in image_library.processed_files:
                    print(f"Skipping existing file: {filename}")
                    continue
                
                try:
                    print(f"Processing new image: {filename}")
                    image = Image.open(img_path)
                    features = process_image(image)
                    if image_library.add_image(img_path_str, features):
                        new_images_added = True
                    current_hashes[img_path_str] = image_library.get_image_hash(img_path_str)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    if new_images_added:
        print("\nNew images added, rebuilding index...")
        image_library.build_index()
        print("Saving cache...")
        image_library.save_cache()
        image_library.save_image_hashes(current_hashes)
        print("Image library updated successfully!")
    else:
        print("\nNo new images to process!")
    
    print(f"\nTotal images in library: {len(image_library.image_paths)}")
    print("=== Image Library Build Completed ===\n")

# 在全局范围内构建图片库
print("Initializing image library...")
build_image_library()
print("Image library initialization completed!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_text', methods=['POST'])
def search_text():
    try:
        text = request.json.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # 处理文本查询
        text_features = process_text(text)
        results = image_library.search(text_features)
        
        # 格式化结果
        formatted_results = [
            {'image': os.path.basename(path), 'score': float(score)}
            for path, score in results
        ]
        
        return jsonify(formatted_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search_image', methods=['POST'])
def search_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        # 读取和处理上传的图片
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_features = process_image(image)
        
        # 搜索相似图片
        results = image_library.search(image_features)
        
        # 格式化结果
        formatted_results = [
            {'image': os.path.basename(path), 'score': float(score)}
            for path, score in results
        ]
        
        return jsonify(formatted_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    try:
        # 确保文件名不包含路径分隔符
        filename = os.path.basename(filename)
        
        if STORAGE_TYPE == 's3':
            # S3 模式：生成预签名 URL
            url = get_s3_presigned_url(filename)
            if url:
                return redirect(url)
            else:
                return "Failed to generate S3 URL", 500
        else:
            # 本地模式：直接从文件系统提供图片
            full_path = os.path.join(os.path.abspath(IMAGE_LIBRARY_DIR), filename)
            print(f"Attempting to serve image: {full_path}")
            
            if not os.path.exists(full_path):
                print(f"File not found: {full_path}")
                return "Image not found", 404
                
            if not os.access(full_path, os.R_OK):
                print(f"No read permission for file: {full_path}")
                return "Permission denied", 403
                
            file_stat = os.stat(full_path)
            print(f"File stats - size: {file_stat.st_size}, permissions: {oct(file_stat.st_mode)}")
            
            return send_from_directory(
                IMAGE_LIBRARY_DIR,
                filename,
                mimetype='image/jpeg'
            )
    except Exception as e:
        print(f"Error serving image {filename}: {str(e)}")
        return f"Error serving image: {str(e)}", 500

# 添加一个路由来检查图片是否存在
@app.route('/check_image/<path:filename>')
def check_image(filename):
    try:
        filename = os.path.basename(filename)
        if STORAGE_TYPE == 's3':
            try:
                # 检查文件是否存在于 S3
                s3_client.head_object(Bucket=S3_BUCKET, Key=filename)
                return jsonify({
                    'exists': True,
                    'filename': filename
                })
            except:
                return jsonify({
                    'exists': False,
                    'filename': filename
                })
        else:
            # 本地模式
            full_path = os.path.join(IMAGE_LIBRARY_DIR, filename)
            exists = os.path.isfile(full_path)
            return jsonify({
                'exists': exists,
                'full_path': full_path,
                'filename': filename
            })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # 确保缓存目录存在
    os.makedirs('cache', exist_ok=True)
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000) 