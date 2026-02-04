@echo off
cls
echo =====================================
echo         YOLOv8 一键部署工具
echo =====================================
echo 1. 配置运行环境
echo 2. 自动运行测试脚本
echo =====================================
echo.

:: 步骤1：配置环境
echo 🔧 开始配置运行环境...
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ultralytics==8.4.11 torch==2.3.0 numpy opencv-python pyyaml requests -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
python -c "import os; [os.makedirs(d, exist_ok=True) for d in ['models/trained', 'data/datasets', 'data/test_images', 'runs/test']]"

:: 步骤2：运行测试脚本
echo.
echo ✅ 环境配置完成！
echo.
echo 🚀 开始运行YOLOv8测试脚本...
if exist "scripts/test/test_detect.py" (
    python scripts/test/test_detect.py
) else (
    echo ⚠️  未找到测试脚本，自动创建简易版本...
    echo from ultralytics import YOLO>scripts/test/test_detect.py
    echo model = YOLO('yolov8s.pt')>>scripts/test/test_detect.py
    echo results = model('https://ultralytics.com/images/bus.jpg', save=True)>>scripts/test/test_detect.py
    echo print("\n✅ 简易测试完成！结果保存在 runs/detect/ 目录")>>scripts/test/test_detect.py
    python scripts/test/test_detect.py
)

:: 步骤3：完成提示
echo.
echo =====================================
echo 🎉 部署+测试全流程完成！
echo 📌 后续操作：
echo    1. 训练模型：python scripts/train/train_custom.py
echo    2. 查看测试结果：runs/test/detect_result/bus.jpg
echo =====================================
pause
