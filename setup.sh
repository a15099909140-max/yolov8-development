#!/bin/bash
# 一键配置YOLOv8环境（服务器/Linux/Mac）

# 升级pip
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 自动下载YOLOv8预训练模型
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

echo "✅ 环境配置完成！训练命令：python scripts/train/train_custom.py"
