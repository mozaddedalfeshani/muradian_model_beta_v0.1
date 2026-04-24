from dataclasses import dataclass

@dataclass
class MuradianConfig:
    # Model architecture
    vocab_size: int = 16384 # Scaled up from 8k
    context_length: int = 512
    n_layer: int = 12       # Increased from 8
    n_head: int = 12        # Increased from 10
    n_embd: int = 384       # Increased from 320
    dropout: float = 0.1
    bias: bool = False
    
    # Training
    batch_size: int = 4
    learning_rate: float = 5e-4 # Slightly lower for larger model
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # LR Scheduler
    decay_lr: bool = True
    warmup_iters: int = 200
    lr_decay_iters: int = 2000
    min_lr: float = 5e-5
    
    # System
    device: str = 'cuda'
    dtype: str = 'float16'
    compile: bool = False
