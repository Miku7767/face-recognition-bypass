# 人脸识别认证绕过技术研究 — 第一阶段

> 省级大创项目：光学显示式人脸识别认证绕过技术研究  
> 导师：幸玮老师  
> 任务1：ArcFace跑通与人脸识别基线建立

## 项目概述

本项目研究通过屏幕翻拍、投影、打印照片等光学显示方式，对人脸识别认证系统进行静态欺骗攻击，评估其安全性。

第一阶段目标：使用 ArcFace 预训练模型搭建 1:N 人脸识别系统，在 LFW 标准数据集上建立攻击前的准确率基线。

## 核心原理

ArcFace 不直接比较两张照片的像素，而是将人脸"压缩"为一个 512 维特征向量（embedding）。同一人的向量在高维空间中方向接近（余弦相似度高），不同人的方向远离（余弦相似度低）。本项目的攻击研究本质上是研究：**攻击样本能否骗过这个向量比对的决策边界**。

## 模型信息

- **模型**：InsightFace `buffalo_l`（ArcFace + ResNet50）
- **训练数据**：MS-Celeb-1M（约1000万张图像）
- **模型文件**：5个 ONNX 文件，共约 275MB，存放于 `~/.insightface/models/buffalo_l/`

| 文件 | 功能 |
|------|------|
| `det_10g.onnx` | 人脸检测（SCRFD） |
| `w600k_r50.onnx` | ArcFace 识别（512维特征向量） |
| `2d106det.onnx` | 106点2D关键点 |
| `1k3d68.onnx` | 68点3D关键点 |
| `genderage.onnx` | 性别年龄估计 |

## 环境搭建

### 前置要求

- Python 3.11（Python 3.14 暂不支持 ML 包）
- Windows / Linux / macOS
- 无需 GPU（CPU 推理即可）

### 安装步骤

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活环境
venv\Scripts\activate        # Windows
source venv/bin/activate      # Linux/macOS

# 3. 安装依赖
pip install insightface opencv-python numpy scipy onnxruntime

# 4. 下载预训练模型（首次运行自动下载，约 275MB）
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=-1)"
```

> 国内用户可能遇到 GitHub 下载慢的问题，可使用加速器或手动下载后放入 `~/.insightface/models/buffalo_l/`。

## 使用方式

### 1:N 人脸识别

```bash
# 1. 准备数据
#    data/gallery/张三/001.jpg   ← 人脸库：按人建子文件夹
#    data/query/待识别.jpg       ← 待识别照片

# 2. 运行
python src/face_recognizer.py
```

### Python API

```python
from src.face_recognizer import FaceRecognizer

fr = FaceRecognizer()                        # 加载模型
fr.build_gallery("data/gallery")              # 构建人脸库
person, sim = fr.identify("data/query/test.jpg")  # 1:N 识别
print(person, sim)                            # 输出：周家名 0.741
```

### LFW 标准评估

```bash
python src/face_recognizer.py evaluate-lfw --lfw-dir data/lfw --pairs data/lfw/pairs.txt
```

## 实测验证结果

| 场景 | 相似度 | 判定 | 结果 |
|------|--------|------|------|
| 同一个人（周家名） | 0.741 | ≥0.65 匹配 | ✓ 正确 |
| 完全陌生人 | 0.080 | <0.65 拒识 | ✓ 正确 |
| 边界案例 | 0.638 | <0.65 拒识 | ✓ 正确 |

## 阈值分析

阈值直接影响识别准确率。本次实验中，若使用 0.25 作为阈值（组员肖宇通过阈值分析得到的最优值），LFW 准确率可达 98.23%。若使用默认值 0.65，同人对可能被误判。**建议**在实际应用中通过 ROC 曲线分析选择最优阈值。

## 踩坑记录

| 问题 | 原因 | 解决 |
|------|------|------|
| Python 3.14 装不上 ML 包 | 版本太新 | 降级到 Python 3.11 |
| `cv2.imread` 读不了中文路径 | Windows ANSI 编码限制 | 改用 `np.fromfile` + `cv2.imdecode` |
| GitHub 模型下载极慢 | 国内网络 | 加速器 + 断点续传 |
| zip 文件多次中断后损坏 | 残留文件污染 | 删除后重新完整下载 |
| ONNX 模型文件路径不对 | zip 扁平结构 | 手动创建子目录并移入 |

## 文件结构

```
大创项目/
├── .gitignore              # Git 忽略规则
├── README.md               # 本文档
├── src/
│   └── face_recognizer.py  # ArcFace 识别模块（核心）
├── data/
│   ├── gallery/            # 人脸库（需自行准备）
│   ├── query/              # 待识别照片（需自行准备）
│   └── lfw/                # LFW 数据集（需自行下载）
└── outputs/                # 输出结果
```

## 后续阶段

- **任务2**：MTCNN 归一化模型优化
- **任务3**：商用 API 测试（腾讯云/阿里云/百度云/Face++）
- **任务4**：其他人脸识别模型筛选（MagFace、AdaFace 等）
- **任务5**：非直接摄像头拍摄数据集收集与攻击实验
