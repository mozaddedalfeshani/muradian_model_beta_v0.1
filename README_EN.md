# MiniGPT Beta v0.1

[![Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/mozaddedalfeshani/muradian_model_beta_v0.1)

This is a custom **Decoder-Only Transformer** language model built from scratch using **PyTorch**. The main goal of this project is to create a small but powerful model that can generate text based on specific data (e.g., personal data).

## 🚀 How it works

The model follows a **Generative Pre-trained Transformer (GPT)** architecture. It takes some text as input and predicts what the next most likely word (Token) could be.

### Workflow:
1. **Data Preparation:** Text data is collected and processed using `prepare_personal_data.py` or your own scripts.
2. **Tokenization:** Text cannot be fed directly into the model, so it is divided into small tokens using BPE (Byte Pair Encoding).
3. **Training:** The model is trained on this data using `train.py` so it can perform text prediction and generation.
4. **Generation:** Once training is complete, we can test the model's output with `generate.py`.

## 🛠 Methodology

Several advanced modern deep learning methods were used in the development of this model:

*   **Transformer Architecture:** A transformer block with 8 layers and 10 attention heads was used.
*   **BPE Tokenization:** A custom tokenizer with a vocab size of 8192 was created using the HuggingFace `tokenizers` library.
*   **Flash Attention:** PyTorch's scaled dot product attention was used to speed up processing.
*   **Optimization:**
    *   **AdamW Optimizer:** Used for updating model weights.
    *   **Cosine LR Decay:** This scheduler was used to gradually decrease the learning rate during training.
    *   **Weight Tying:** Weights of the input embedding and output layers were shared to keep the number of parameters low and improve performance.
*   **Weight Normalization:** Layer Normalization (LayerNorm) and Dropout were used to prevent overfitting.

## 📊 Technical Specs

- **Parameters:** ~15 Million (15M)
- **Layers:** 8
- **Hidden Size:** 320
- **Heads:** 10
- **Context Length:** 512 tokens
- **Vocabulary Size:** 8192

## 📂 Project Structure

- `model.py`: Core code for the Transformer architecture.
- `config.py`: Hyper-parameters and configuration.
- `tokenizer.py`: BPE tokenizer handler.
- `dataset.py`: Data loading pipeline.
- `train.py`: Training script.
- `generate.py`: Text generation or chat script.
- `prepare_personal_data.py`: Custom data processing tool.
- `export_onnx.py`: Tool to export the model to ONNX format.

## ⚙️ Setup & Usage

### 1. Environment Setup
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preparation
To create a training set from your own text data:
```bash
python prepare_personal_data.py
```

### 3. Start Training
```bash
python train.py
```

### 4. Using the Model
```bash
python generate.py --prompt "User: Hello!\nAssistant:" --tokens 100
```

## 📝 Note
This project is an experimental work and is being continuously improved. It was developed to experiment with AI and language models.

---
**Repository:** [https://github.com/mozaddedalfeshani/muradian_model_beta_v0.1](https://github.com/mozaddedalfeshani/muradian_model_beta_v0.1)
