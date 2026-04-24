# Muradian Model Beta v0.1

[![Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/mozaddedalfeshani/muradian_model_beta_v0.1)

![Muradian Model Workflow Sketch](docs/workflow_sketch.png)

This is a custom **Decoder-Only Transformer** language model built from scratch using **PyTorch**. The main goal of this project is to create a small but powerful model that can generate text based on specific data (like Bangla chat data).

## 🚀 How it works

The model follows a **Generative Pre-trained Transformer (GPT)** architecture. It takes some text as input and predicts what the next most likely word (Token) could be.

### Workflow:
1. **Data Preparation:** Use `prepare_huggingface_data.py` to collect and process the Bangla Alpaca dataset from Hugging Face.
2. **Tokenization:** Text is divided into 16,384 small tokens using BPE (Byte Pair Encoding). Use `train_tokenizer.py` for this.
3. **Training:** The model is trained on this data using `train.py`. This now includes a validation loop and live sample generation.
4. **Generation & Testing:** Once training is complete, we can test the model's output with `generate.py` or `test_questions.py`.

### 📂 Data Processing Pipeline

A non-coder friendly view of how our data is processed step-by-step:

![Muradian Data Pipeline](docs/data_pipeline_flow.png)

### 📊 Architecture Diagram

![Muradian Model Architecture](docs/architecture_whiteboard.png)

## 🛠 Methodology

Several advanced modern deep learning methods were used in the development of this model:

*   **Transformer Architecture:** A transformer block with 12 layers and 12 attention heads was used.
*   **BPE Tokenization:** A custom tokenizer with a vocab size of 16,384 was created using the HuggingFace `tokenizers` library.
*   **Flash Attention:** PyTorch's scaled dot product attention was used to speed up processing.
*   **Training & Evaluation:**
    *   **Validation Loop:** Tracks model performance during training by checking validation loss.
    *   **AdamW Optimizer:** Used for updating model weights.
    *   **Cosine LR Decay:** Gradually decreases the learning rate during training.
*   **Advanced Features:**
    *   **Weight Tying:** Shared weights between input embedding and output layers.
    *   **Clean Dataset Pipeline:** Ensured data cleaning and proper User/Assistant formatting.

## 📊 Technical Specs

- **Parameters:** ~27.5 Million (27.5M)
- **Layers:** 12
- **Hidden Size:** 384
- **Heads:** 12
- **Context Length:** 512 tokens
- **Vocabulary Size:** 16,384

## 📂 Project Structure

- `model.py`: Core code for the Transformer architecture.
- `config.py`: Hyper-parameters and configuration.
- `tokenizer.py`: BPE tokenizer handler.
- `dataset.py`: Data loading pipeline.
- `train.py`: Training script (with Eval loop).
- `generate.py`: Text generation script.
- `train_tokenizer.py`: Tokenizer training tool.
- `prepare_huggingface_data.py`: Bangla chat data processing tool.
- `test_questions.py`: Script for testing the model with specific questions.
- `export_onnx.py`: Tool to export the model to ONNX format.

## ⚙️ Setup & Usage

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python prepare_huggingface_data.py
python train_tokenizer.py
```

### 3. Start Training
```bash
python train.py
```

### 4. Using the Model
```bash
python generate.py --prompt "User: কেমন আছ?\nAssistant:" --tokens 100
# or
python test_questions.py
```

---
**Repository:** [https://github.com/mozaddedalfeshani/muradian_model_beta_v0.1](https://github.com/mozaddedalfeshani/muradian_model_beta_v0.1)
