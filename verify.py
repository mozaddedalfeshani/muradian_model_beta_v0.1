from model import MiniGPT
from config import MiniGPTConfig
import torch

def verify_model():
    config = MiniGPTConfig()
    model = MiniGPT(config)
    
    n_params = model.get_num_params()
    print(f"Total non-embedding parameters: {n_params/1e6:.2f}M")
    
    # Test forward pass
    idx = torch.randint(0, config.vocab_size, (1, 10))
    logits, loss = model(idx)
    print(f"Forward pass successful. Logits shape: {logits.shape}")
    
    # Test with targets
    targets = torch.randint(0, config.vocab_size, (1, 10))
    logits, loss = model(idx, targets)
    print(f"Loss calculation successful. Loss: {loss.item():.4f}")

if __name__ == "__main__":
    verify_model()
