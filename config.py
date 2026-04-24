from dataclasses import dataclass

@dataclass
class MiniGPTConfig:
    # Model architecture
    vocab_size: int = 8192
    context_length: int = 512
    n_layer: int = 8
    n_head: int = 10
    n_embd: int = 320
    dropout: float = 0.1
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    # Training
    batch_size: int = 8
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # LR Scheduler
    decay_lr: bool = True
    warmup_iters: int = 100
    lr_decay_iters: int = 1000
    min_lr: float = 6e-5
    
    # System
    device: str = 'cuda'
    dtype: str = 'float16'  # 'float32', 'bfloat16', or 'float16'
    compile: bool = False  # disable by default to reduce VRAM usage on small GPUs
