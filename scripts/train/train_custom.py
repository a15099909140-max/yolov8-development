from ultralytics import YOLO
import os

# -------------------------- 仅需修改这里 --------------------------
data_path = "data/datasets/coco128.yaml"  # 数据集配置文件路径
epochs = 50  # 训练轮数
batch_size = 16  # 批次大小（根据GPU调整）
# ------------------------------------------------------------------

# 确保保存目录存在
os.makedirs("models/trained", exist_ok=True)

# 加载模型+训练
model = YOLO("yolov8s.pt")
print(f"🚀 开始训练（数据集：{data_path}）")
results = model.train(
    data=data_path,
    epochs=epochs,
    batch=batch_size,
    imgsz=640,
    device=0,
    project="models/trained"
)
print(f"✅ 训练完成！权重存于：models/trained")
