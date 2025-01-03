# QF_Clip 图像检索系统

这是一个基于MobileCLIP模型的图像检索系统，可以通过文本或图片来检索相似的图片。

## 功能特点

- 支持使用文本描述检索相似图片
- 支持使用图片检索相似图片
- 本地图片库管理和检索
- 高效的向量索引和检索

## 目录结构

```
QF_Clip/
├── checkpoints/        # 模型文件目录
│   └── mobileclip_s0.pt
├── docs/              # 文档和测试图片
├── images/            # 图片库目录
└── notebook/          # Jupyter notebooks
    └── clip_demo.ipynb
```

## 环境要求

- Python 3.10+
- PyTorch
- MobileCLIP
- FAISS
- Pillow
- NumPy
- Matplotlib

## 使用说明

1. 准备环境：
   ```bash
   pip install torch torchvision faiss-cpu pillow numpy matplotlib tqdm
   ```

2. 下载MobileCLIP模型：
   - 从官方仓库下载模型文件：https://github.com/apple/ml-mobileclip
   - 将模型文件放置在 `checkpoints` 目录下

3. 准备图片库：
   - 将需要建立索引的图片放入 `images` 目录
   - 支持的图片格式：jpg, jpeg, png

4. 运行演示notebook：
   - 打开 `notebook/clip_demo.ipynb`
   - 按照notebook中的说明逐步执行

## 主要功能说明

### 1. 图片库建立
```python
# 初始化图片库
image_library = ImageLibrary()

# 构建图片库索引
build_image_library("path/to/image/directory")
```

### 2. 文本检索
```python
# 使用文本检索图片
query_text = "a cat"
results = image_library.search(process_text(query_text))
```

### 3. 图片检索
```python
# 使用图片检索相似图片
query_image_path = "path/to/query/image.jpg"
results = image_library.search(process_image(query_image_path))
```

## 注意事项

1. 图片处理：
   - 所有图片会被自动转换为RGB格式
   - 建议使用清晰的图片以获得更好的检索效果

2. 性能优化：
   - 使用FAISS进行高效的向量检索
   - 图片特征向量会被缓存以提高检索速度

3. 内存使用：
   - 图片库大小会影响内存使用
   - 建议根据实际需求调整检索结果数量

## 后续开发计划

1. 添加Web界面支持
2. 优化向量索引性能
3. 添加批量处理功能
4. 支持更多图片格式
5. 添加图片预处理选项