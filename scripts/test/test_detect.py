# YOLOv8 通用测试脚本（自动下载测试数据，CPU/GPU适配）
from ultralytics import YOLO
import os
import requests
import warnings
warnings.filterwarnings('ignore')

# ======================== 可配置参数 ========================
WEIGHTS_PATH = "models/trained/yolov8s_coco128_train/weights/best.pt"  # 训练好的权重路径
TEST_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"              # 自动下载的测试图片
OUTPUT_DIR = "runs/test"                                              # 检测结果保存目录
CONF_THRESHOLD = 0.25                                                 # 置信度阈值
DEVICE = "auto"                                                       # 自动检测设备（cpu/gpu）
# ===========================================================

def download_test_image(save_path):
    """自动下载测试图片"""
    if os.path.exists(save_path):
        print(f"✅ 测试图片已存在：{save_path}")
        return
    
    print(f"📥 正在下载测试图片：{TEST_IMAGE_URL}")
    try:
        response = requests.get(TEST_IMAGE_URL, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ 测试图片下载完成：{save_path}")
    except Exception as e:
        print(f"❌ 图片下载失败：{e}")
        print("💡 备用方案：手动将测试图片放到 data/test_images/bus.jpg")
        raise

def init_environment():
    """初始化测试环境"""
    # 创建必要目录
    dirs = [OUTPUT_DIR, "data/test_images"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # 下载测试图片
    test_img_path = "data/test_images/bus.jpg"
    download_test_image(test_img_path)
    
    # 检查权重文件
    if not os.path.exists(WEIGHTS_PATH):
        print(f"⚠️  未找到训练权重：{WEIGHTS_PATH}")
        print("📥 自动加载预训练权重 yolov8s.pt 进行测试")
        return "yolov8s.pt", test_img_path
    return WEIGHTS_PATH, test_img_path

def main():
    # 1. 初始化环境
    print("🔧 初始化测试环境...")
    weights, test_img = init_environment()
    
    # 2. 加载模型
    print(f"\n📦 加载模型：{weights}")
    model = YOLO(weights)
    
    # 3. 开始检测
    print(f"\n🚀 开始检测图片：{test_img}")
    results = model(
        source=test_img,
        conf=CONF_THRESHOLD,
        device=0,
        save=True,               # 保存检测结果图片
        save_txt=True,           # 保存检测结果txt
        project=OUTPUT_DIR,
        name="detect_result",
        show_labels=True,
        show_conf=True
    )
    
    # 4. 输出检测结果
    print("\n📊 检测结果：")
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls = box.cls
                conf = box.conf
                cls_name = model.names[int(cls)]
                print(f"   🎯 {cls_name} | 置信度：{conf:.2f}")
    
    # 5. 提示结果路径
    result_img_path = f"{OUTPUT_DIR}/detect_result/bus.jpg"
    print(f"\n✅ 检测完成！")
    print(f"📸 结果图片：{result_img_path}")
    print(f"📝 结果标注：{OUTPUT_DIR}/detect_result/labels/bus.txt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 测试脚本执行失败：{e}")
        print("\n💡 快速解决方案：")
        print("   1. 确保已安装依赖：pip install ultralytics requests")
        print("   2. 网络异常时，手动下载图片到 data/test_images/bus.jpg")
        print("   3. 无训练权重时，脚本会自动使用预训练yolov8s.pt测试")
