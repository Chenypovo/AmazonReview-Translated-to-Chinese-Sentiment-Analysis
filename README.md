# Amazon Review Chinese Sentiment Analysis 🎯

**Fine-tuned Qwen3 models for Chinese sentiment classification on Amazon Digital Music reviews**

This project translates ~10,000 Amazon Digital Music reviews (2023) into Chinese using Qwen API, then performs 3-class sentiment analysis via LLaMA Factory.

---

## 📊 Results

| Model | Setting | Accuracy | Latency/ Sample |
| :--- | :---: | :---: | :---: |
| **Qwen3-4B-Instruct** | 10-shot | **82.75%** | 14.52s |
| Qwen3-4B-Instruct | 5-shot | 78.25% | 1.47s |
| Qwen3-4B-Instruct | Zero-shot | 77.25% | 0.52s |
| Qwen3-0.6B-Base | 5-shot | 79.66% | 0.26s |
| Qwen3-0.6B-Base | Zero-shot | 54.35% | 0.06s |

**Key Findings:**
- 4B instruction model achieves best accuracy (82.75%) with 10-shot, but latency is high
- 0.6B base model with 5-shot is a strong candidate (79.66%, 0.26s) for industrial deployment
- LoRA fine-tuning on 0.6B is recommended for balancing performance and cost

---

## 🏗️ Pipeline

```
Raw Data (HuggingFace) → Qwen Translation → Data Augmentation → LLaMA Factory Format → Fine-tuning → Evaluation
```

### Stages

1. **Data Translation**: Concurrent translation using Qwen API
2. **Data Augmentation**: Increase minority class samples
3. **Format Conversion**: Convert CSV to LLaMA Factory `.jsonl` format
4. **Model Training**: Fine-tune Qwen3 models via LLaMA Factory
5. **Evaluation**: Zero-shot and few-shot inference with accuracy/latency metrics

---

## 🛠 Installation

```bash
# Clone the repo
git clone https://github.com/Chenypovo/AmazonReview-Translated-to-Chinese-Sentiment-Analysis.git
cd AmazonReview-Translated-to-Chinese-Sentiment-Analysis

# Create environment
conda create -n llama_env python=3.10 -y
conda activate llama_env

# Install PyTorch (adjust CUDA version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install LLaMA Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,bitsandbytes,webui,api]"
cd ..
```

---

## 🚀 Usage

### Set API Key

```bash
export QWEN_API_KEY="your_api_key_here"
```

### Data Preparation

```bash
# Download dataset
python task1/scripts/data_retrive.py

# Translate to Chinese
python task1/src/translate.py

# Data augmentation
python task1/src/data_aug.py

# Convert to LLaMA Factory format
python task1/prepare_llamafactory_data.py
```

### Training & Evaluation

```bash
# Train baseline model
python task1/src/train_baseline_5class.py

# Evaluate locally
python task1/qwen_eval_local.py

# Smoke test
python task1/qwen_smoke.py
```

---

## 📂 Project Structure

*   `task1/scripts/`: Data retrieval scripts
*   `task1/src/`: Core modules (translation, training, evaluation)
*   `task1/notebook/`: Jupyter notebooks for data exploration and preprocessing
*   `task1/prepare_llamafactory_data.py`: Format conversion script
*   `task1/problems_encountered.md`: Issues and solutions during development
*   `requirements.txt`: Python dependencies

---

## 📝 TODO

- [ ] Add training logs and evaluation reports
- [ ] Provide model weight downloads
- [ ] Add API deployment example
- [ ] Docker configuration
- [ ] Performance benchmark report

---

## 📧 Contact

Developed by **YiPeng Chen**.
- **Email**: zcapy55@ucl.ac.uk
- **GitHub**: [Chenypovo](https://github.com/Chenypovo)

---

## 📄 License

MIT License
