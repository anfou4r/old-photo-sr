# 基于深度学习的老旧照片高清修复系统

本项目实现了基于深度学习的老旧照片超分辨率修复系统，采用 RRDB（Residual-in-Residual Dense Block）网络架构，能够将低分辨率、受损的老旧照片恢复为高清图像。

## 项目特点

- **RRDB 超分辨率网络**：采用 ESRGAN 中的 RRDB 架构，支持 4 倍超分辨率重建
- **老旧照片退化模拟**：模拟噪声、模糊、JPEG 压缩、泛黄、划痕、暗角等老照片常见退化
- **两阶段训练策略**：PSNR 预训练 + GAN 微调，兼顾保真度与感知质量
- **感知损失 + 对抗损失**：使用 VGG19 感知损失和 GAN 损失提升视觉效果
- **分块推理**：支持大图分块处理，避免 GPU 显存溢出
- **Gradio 图形界面**：提供简洁的 Web 操作界面，支持实时预览

## 项目结构

```
old-photo-sr/
├── configs/
│   └── train_config.yaml      # 训练配置文件
├── models/
│   ├── __init__.py
│   ├── rrdbnet.py              # RRDB 超分辨率生成器
│   ├── discriminator.py        # VGG 风格判别器
│   └── losses.py               # 感知损失与 GAN 损失
├── data/
│   ├── __init__.py
│   └── dataset.py              # 数据集加载与预处理
├── utils/
│   ├── __init__.py
│   ├── degradation.py          # 老照片退化模拟
│   ├── metrics.py              # PSNR/SSIM 评估指标
│   └── img_utils.py            # 图像 I/O 工具
├── train.py                    # 训练脚本
├── inference.py                # 推理脚本
├── app.py                      # Gradio GUI 界面
├── prepare_data.py             # 数据集准备工具
├── requirements.txt            # 依赖包列表
└── README.md
```

## 环境安装

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

**主要依赖**：PyTorch >= 1.12、OpenCV、Gradio、TensorBoard

## 数据准备

推荐使用以下公开数据集：

- **DIV2K**：800 张高质量 2K 分辨率训练图像
  - 下载地址：https://data.vision.ee.ethz.ch/cvl/DIV2K/
- **Flickr2K**：2650 张高质量图像
- **Set5/Set14/BSD100**：标准超分辨率测试集

```bash
# 将下载的 HR 图像放入 datasets/all_images/ 目录

# 划分训练/验证集
python prepare_data.py split --input datasets/all_images --ratio 0.9

# （可选）裁剪为小块以加速训练
python prepare_data.py patch --input datasets/train --output datasets/train_patches
```

## 训练模型

### 第一阶段：PSNR 预训练

使用 L1 像素损失进行预训练，获得基础超分辨率能力：

```bash
python train.py --config configs/train_config.yaml --stage psnr
```

### 第二阶段：GAN 微调

在预训练模型基础上，使用感知损失和对抗损失进行微调，提升视觉质量：

```bash
python train.py --config configs/train_config.yaml --stage gan \
    --pretrain checkpoints/best_psnr_model.pth
```

训练过程可通过 TensorBoard 监控：

```bash
tensorboard --logdir logs/
```

## 推理测试

```bash
# 单张图像
python inference.py --input test.jpg --output results/ \
    --model checkpoints/best_model.pth

# 批量处理
python inference.py --input inputs/ --output results/ \
    --model checkpoints/best_model.pth

# 大图分块处理
python inference.py --input inputs/ --output results/ \
    --model checkpoints/best_model.pth --tile_size 512
```

## 图形界面

```bash
python app.py --model checkpoints/best_model.pth --port 7860
```

启动后在浏览器中打开 `http://localhost:7860`，即可上传照片进行修复。

## 网络架构

### RRDB 生成器

```
输入 (3, H, W) -> Conv -> [RRDB x 23] -> Conv -> ↑2x -> ↑2x -> Conv -> Conv -> 输出 (3, 4H, 4W)
```

- **Residual Dense Block (RDB)**：5 层密集连接卷积 + 残差缩放
- **RRDB**：3 个 RDB 级联 + 残差连接
- **上采样**：最近邻插值 + 卷积，逐步 2 倍放大

### 损失函数

| 损失 | 作用 | 权重 |
|------|------|------|
| L1 像素损失 | 保证像素级保真度 | 1.0 |
| VGG 感知损失 | 提升感知质量 | 1.0 |
| GAN 对抗损失 | 生成更自然的纹理 | 0.1 |

### 老照片退化模拟

训练时通过模拟以下退化增强模型对老照片的鲁棒性：

- 高斯噪声 / 高斯模糊
- JPEG 压缩伪影
- 色彩褪色 / 泛黄（棕褐色调）
- 划痕损伤
- 暗角效果

## 评估指标

- **PSNR**（峰值信噪比）：衡量像素级重建精度
- **SSIM**（结构相似性）：衡量结构信息保持程度

## 配置说明

主要训练参数在 `configs/train_config.yaml` 中配置：

- `model.num_block`：RRDB 块数量（默认 23，减少可加速但降低质量）
- `model.num_feat`：特征通道数（默认 64）
- `dataset.train.hr_size`：训练时 HR 裁剪大小
- `train.total_epochs`：总训练轮数
- `train.lr_g`：生成器学习率
- `degradation`：退化模拟参数

## 参考文献

1. Wang X, et al. "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks." ECCV 2018.
2. Wang X, et al. "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data." ICCVW 2021.
3. Ledig C, et al. "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network." CVPR 2017.
