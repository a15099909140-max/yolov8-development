# YOLOv8开发项目（开箱即用）

## 快速开始
1. 拉取代码：
   ```bash
   git clone https://github.com/a15099909140-max/yolov8-development.git
   cd yolov8-development
   ```

2. 一键配置环境：
   - Linux/服务器：`bash setup.sh`
   - Windows：双击`setup.bat`

3. 开始训练：
   ```bash
   python scripts/train/train_custom.py
   ```


## 目录结构
```
yolov8-development/
├── libs/          # YOLOv8源码
├── data/          # 数据集目录（自行添加）
├── models/        # 训练权重
├── scripts/       # 训练/推理脚本
├── setup.sh/bat   # 一键环境配置
├── requirements.txt # 依赖清单
└── README.md      # 使用指南
```
