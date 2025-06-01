# 音乐生成项目

这个项目实现了一个基于Transformer的音乐生成模型，可以根据和弦生成对应的旋律。

## 项目结构

```
Assign2/
├── data/               # 数据文件夹
│   ├── raw/           # 原始MIDI文件
│   └── processed/     # 处理后的数据
├── models/            # 保存训练好的模型
├── src/              # 源代码
│   ├── data/         # 数据处理相关代码
│   ├── models/       # 模型定义
│   └── utils/        # 工具函数
├── utils/            # 通用工具函数
├── notebooks/        # Jupyter notebooks
├── requirements.txt  # 项目依赖
└── README.md        # 项目说明
```

## 安装

1. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 数据准备：
```bash
python src/data/prepare_data.py
```

2. 训练模型：
```bash
python src/train.py
```

3. 生成音乐：
```bash
python src/generate.py
```

## 项目特点

- 基于Transformer架构
- 支持和弦到旋律的生成
- 使用MAESTRO数据集
- 包含完整的训练和生成流程

## 注意事项

- 确保有足够的磁盘空间存储数据集
- 推荐使用GPU进行训练
- 可以根据需要调整模型参数 