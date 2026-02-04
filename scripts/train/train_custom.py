# YOLOv8基础训练脚本（修复数据集路径问题）
from ultralytics import YOLO
import os
from ultralytics.utils.downloads import download

# -------------------------- 仅需修改这里 --------------------------
epochs = 3  # 测试用小轮数，正式训练改50
batch_size = 8  # CPU训练建议减小批次，避免内存不足
# ------------------------------------------------------------------

# 确保模型保存目录存在
os.makedirs("models/trained", exist_ok=True)

# 加载官方预训练模型
model = YOLO("yolov8s.pt")

# 关键：使用YOLOv8内置的coco128数据集（自动下载到正确路径）
data_path = "coco128.yaml"  # 直接用内置配置，自动下载数据集

# 开始训练（CPU模式，适配无GPU环境）
print(f"🚀 开始训练（数据集：{data_path}，轮数：{epochs}）")
results = model.train(
    data=data_path,          # 内置数据集配置，自动下载
    epochs=epochs,           # 训练轮数
    batch=batch_size,        # 批次大小（CPU建议8/4）
    imgsz=640,               # 图片尺寸
    device="cpu",            # CPU训练（有GPU后改0）
    project="models/trained",# 权重保存路径
    name="yolov8s_first_train",  # 实验名称
    save=True                # 保存权重
)

print(f"✅ 训练完成！权重保存在：models/trained/yolov8s_first_train")
