# MiniGPT: 15M Parameter Decoder-Only Transformer

This project implements a small GPT-style language model from scratch using PyTorch. It is designed for educational purposes and can be trained on custom datasets.

## Architecture Specs
- **Parameters**: ~15M
- **Layers**: 8
- **Hidden Size**: 320
- **Heads**: 10
- **Context Length**: 512
- **Vocab Size**: 8192 (BPE)

## Project Structure
- `config.py`: Hyperparameters and model configuration.
- `model.py`: Transformer architecture implementation.
- `tokenizer.py`: BPE tokenizer (using HuggingFace `tokenizers`).
- `dataset.py`: PyTorch dataset and dataloader.
- `train.py`: Full training pipeline.
- `generate.py`: Inference script with sampling (top-k, top-p, temperature).
- `export_onnx.py`: Script to export the model to ONNX format.

## Setup
1. A virtual environment `.venv` has been created.
2. Dependencies are listed in `requirements.txt`.

To activate the environment:
```bash
source .venv/bin/activate
```

## How to Train
1. Prepare your training data in `data/train.txt`. (A sample is already provided).
2. Train the model:
```bash
python train.py
```
This will:
- Train a BPE tokenizer on your data (if `tokenizer.json` doesn't exist).
- Initialize the model.
- Run the training loop and save checkpoints to `ckpt.pt` and the final model to `model.pt`.

## How to Generate
Once you have `model.pt` and `tokenizer.json`:
```bash
python generate.py --prompt "User: Hello!\nAssistant:" --tokens 50 --temp 0.8
```

## Export to ONNX
To export the trained model:
```bash
python export_onnx.py
```

## Note on Python Version
This project was initialized using the available Python version on the system (Python 3.14). If you specifically require Python 3.10, ensure it is installed and recreate the `.venv` with:
```bash
python3.10 -m venv .venv
```
