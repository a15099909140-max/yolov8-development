#!/bin/bash
# YOLOv8 一键环境配置+测试脚本（Linux/服务器）
set -e  # 出错立即退出
clear

echo "====================================="
echo "        YOLOv8 一键部署工具"
echo "====================================="
echo "1. 配置运行环境"
echo "2. 自动运行测试脚本"
echo "====================================="


# 步骤1：配置环境
echo -e "\n🔧 开始配置运行环境..."

# 升级pip
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装核心依赖
pip install ultralytics==8.4.11 torch==2.3.0 numpy opencv-python pyyaml requests -i https://pypi.tuna.tsinghua.edu.cn/simple

# 预下载YOLOv8s模型
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

# 初始化必要目录
python -c "import os; [os.makedirs(d, exist_ok=True) for d in ['models/trained', 'data/datasets', 'data/test_images', 'runs/test']]"


# 步骤2：自动运行测试脚本
echo -e "\n✅ 环境配置完成！"
echo -e "\n🚀 开始运行YOLOv8测试脚本..."

# 检查测试脚本是否存在
if [ -f "scripts/test/test_detect.py" ]; then
    python scripts/test/test_detect.py
else
    echo "⚠️  未找到测试脚本 scripts/test/test_detect.py"
    echo "💡 正在自动创建简易测试脚本..."
    # 自动生成备用测试脚本
    cat > scripts/test/test_detect.py << EOF
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
results = model('https://ultralytics.com/images/bus.jpg', save=True)
print("\n✅ 简易测试完成！结果保存在 runs/detect/ 目录")
EOF
    python scripts/test/test_detect.py
fi


# 步骤3：输出完成提示
echo -e "\n====================================="
echo "🎉 部署+测试全流程完成！"
echo "📌 后续操作："
echo "   1. 训练模型：python scripts/train/train_custom.py"
echo "   2. 查看测试结果：runs/test/detect_result/bus.jpg"
echo "====================================="
