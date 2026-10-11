# UAV Deep Learning

基于深度学习的无人机（UAV）项目。

## 环境配置

项目使用 conda 管理环境，配置见 [`environment.yml`](environment.yml)。

```bash
# 1. 根据配方重建环境（会创建名为 torch 的环境）
conda env create -f environment.yml

# 2. 激活
conda activate torch
```

环境核心组件：

| 组件 | 版本 |
|---|---|
| Python | 3.10 |
| PyTorch | 2.14.1+cu130 |
| torchvision | 0.29.1+cu130 |
| torchaudio | 2.11.0+cu130 |
| CUDA | 13.0 |

> 说明：`environment.yml` 只是环境的"配方"，不含任何二进制文件。
> 实际的 conda 环境（约 5.6 GB）不会、也不应该提交到仓库。

## 验证 GPU 可用

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望输出包含 `True` 和你的 GPU 型号。

## 目录结构

```
.
├── environment.yml   # conda 环境配方
├── .gitignore        # 排除权重、数据集、缓存等大文件
└── README.md
```

## 注意

本项目**不提交**以下内容（已在 `.gitignore` 中排除）：

- 模型权重：`*.pth`、`*.pt`、`*.ckpt`、`*.onnx`
- 数据集：`data/`、`datasets/`
- 训练输出：`runs/`、`outputs/`、`logs/`
- IDE 配置：`.idea/`、`.vscode/`
