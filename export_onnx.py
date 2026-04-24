import torch
from model import MiniGPT
from config import MiniGPTConfig

def export_to_onnx(model_path, onnx_path="model.onnx"):
    # Load config (assuming default for now)
    config = MiniGPTConfig()
    # Note: In a real scenario, you'd load the vocab_size from the trained tokenizer
    
    model = MiniGPT(config)
    # Load state dict
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Clean state dict if needed
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.eval()

    # Create dummy input
    batch_size = 1
    seq_length = 10
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output', 'loss'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    import os
    if os.path.exists("model.pt"):
        export_to_onnx("model.pt")
    else:
        print("model.pt not found. Train the model first.")
