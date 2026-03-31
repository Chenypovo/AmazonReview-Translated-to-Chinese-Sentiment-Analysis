# 🎵 Amazon Review Chinese Sentiment Analysis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Qwen](https://img.shields.io/badge/Model-Qwen3-purple.svg)](https://qwenlm.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**基于 Qwen 翻译 + LLaMA Factory 微调的亚马逊音乐评论中文情感分析系统**

---

## 📖 目录

- [项目简介](#-项目简介)
- [核心成果](#-核心成果)
- [技术架构](#-技术架构)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [详细文档](#-详细文档)
- [引用](#-引用)

---

## 🌟 项目简介

本项目对 2023 年亚马逊数字音乐评论（Amazon Reviews 2023 - Digital Music）进行：

1. **中文翻译** - 使用 Qwen API 将约 10,000 条英文评论翻译为中文
2. **情感分析** - 基于 LLaMA Factory 进行 3 类情感分类（正面/中性/负面）

### 为什么做这个？

- 🌏 中文 NLP 研究需要高质量的中文情感数据集
- 🎯 探索小模型（0.6B）vs 中等模型（4B）在 few-shot 场景下的表现平衡
- ⚡ 研究推理延迟与准确率的 trade-off，为工业部署提供参考

---

## 📊 核心成果

| 模型 | 设置 | 准确率 | 延迟/样本 | 备注 |
|------|------|--------|-----------|------|
| **Qwen3-4B-Instruct** | 10-shot | **82.75%** | 14.52s | 最佳准确率 |
| Qwen3-4B-Instruct | 5-shot | 78.25% | 1.47s | 平衡选择 |
| Qwen3-4B-Instruct | Zero-shot | 77.25% | 0.52s | 最快推理 |
| Qwen3-0.6B-Base | 5-shot | 79.66% | 0.26s | 小模型黑马 |
| Qwen3-0.6B-Base | Zero-shot | 54.35% | 0.06s | 基线参考 |

### 关键发现

- ✅ **4B 指令模型**在 10-shot 下达到最佳结果（82.75%），但延迟较高
- ✅ **0.6B 小模型**在 5-shot 下表现惊人（79.66%），延迟仅 0.26s
- 💡 推荐方案：对 0.6B 进行 LoRA 微调，平衡性能与部署成本

---

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    数据流水线 (Data Pipeline)                │
├─────────────────────────────────────────────────────────────┤
│  原始数据 (HuggingFace) → Qwen 翻译 → 数据增强 → 格式转换    │
│       ↓                    ↓              ↓              ↓   │
│   Amazon-Reviews     英→中翻译      少数类增强    LLaMA-Factory │
│   2023 Digital       (并发处理)    (minority)     (.jsonl)    │
│   Music (EN)                                          ↓       │
│                                                  模型训练      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    模型评估 (Evaluation)                     │
├─────────────────────────────────────────────────────────────┤
│  Qwen3-4B-Instruct  │  Qwen3-0.6B-Base  │  对比分析          │
│  - Zero/5/10-shot   │  - Zero/5-shot    │  - 准确率 vs 延迟  │
│  - 78.25% ~ 82.75%  │  - 54.35% ~ 79.66%│  - 工业部署建议    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.7+ (GPU 加速)
- 8GB+ VRAM (4B 模型), 4GB+ VRAM (0.6B 模型)

### 1️⃣ 安装依赖

```bash
# 创建虚拟环境
conda create -n llama_env python=3.10 -y
conda activate llama_env

# 安装 PyTorch (根据你的 CUDA 版本调整)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt

# 安装 LLaMA Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,bitsandbytes,webui,api]"
cd ..
```

### 2️⃣ 配置 API Key

```bash
# 设置 Qwen API Key
export QWEN_API_KEY="your_api_key_here"

# 或添加到 .env 文件
echo "QWEN_API_KEY=your_api_key_here" > .env
```

### 3️⃣ 数据准备

```bash
# 下载数据集 (HuggingFace)
python task1/scripts/data_retrive.py

# 翻译为中文
python task1/src/translate.py

# 数据增强 (少数类)
python task1/src/data_aug.py

# 转换为 LLaMA Factory 格式
python task1/prepare_llamafactory_data.py
```

### 4️⃣ 训练与评估

```bash
# 训练基线模型
python task1/src/train_baseline_5class.py

# 本地评估
python task1/qwen_eval_local.py

# 烟雾测试
python task1/qwen_smoke.py
```

---

## 📁 项目结构

```
AmazonReview-Chinese-Sentiment-Analysis/
├── README.md                      # 项目文档
├── requirements.txt               # Python 依赖
├── TRAINING_DATASET_FINAL.csv     # 训练数据集
├── task1/
│   ├── scripts/
│   │   └── data_retrive.py        # 数据下载脚本
│   ├── src/
│   │   ├── translate.py           # 翻译模块
│   │   ├── data_aug.py            # 数据增强
│   │   ├── train.py               # 训练脚本
│   │   ├── evaluate_local_llm.py  # 本地评估
│   │   ├── model.py               # 模型定义
│   │   ├── dataset.py             # 数据集处理
│   │   └── loss.py                # 损失函数
│   ├── notebook/
│   │   ├── data_aug.ipynb         # 数据增强 Notebook
│   │   ├── data_aug_clean.ipynb   # 数据清洗 Notebook
│   │   └── llama_fac.ipynb        # LLaMA Factory 格式转换
│   ├── prepare_llamafactory_data.py
│   ├── qwen_eval_local.py         # Qwen 评估脚本
│   ├── qwen_smoke.py              # 烟雾测试
│   └── problems_encountered.md    # 问题记录
└── .gitignore
```

---

## 📚 详细文档

| 文档 | 说明 |
|------|------|
| [problems_encountered.md](./task1/problems_encountered.md) | 开发过程中遇到的问题及解决方案 |
| [notebook/](./task1/notebook/) | Jupyter Notebooks 用于数据探索和预处理 |

---

## 📝 待办事项

- [ ] 完善训练日志和评估报告
- [ ] 添加模型权重下载链接
- [ ] 补充 API 部署示例
- [ ] 添加 Docker 配置
- [ ] 性能基准测试报告

---

## 🙏 致谢

- 数据集：[Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) by McAuley-Lab
- 模型：[Qwen3](https://qwenlm.github.io/) by Alibaba
- 训练框架：[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) by hiyouga

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">

**如果觉得有用，请给个 ⭐ Star！**

[📧 联系我](mailto:zcapy55@ucl.ac.uk) | [🌐 GitHub](https://github.com/Chenypovo)

</div>
