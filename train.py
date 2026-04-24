import os
import time
import math
import torch
from model import MiniGPT
from config import MiniGPTConfig
from tokenizer import BPETokenizer
from dataset import get_dataloader

def train():
    config = MiniGPTConfig()
    
    # Setup device
    device = config.device
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    # Set seed
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Initialize tokenizer and load/train it
    # For this script, we assume tokenizer.json exists or we train a new one
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    if os.path.exists("tokenizer.json"):
        tokenizer.load("tokenizer.json")
    else:
        # If no tokenizer, we might need a small data file to train it
        print("Tokenizer not found. Please run tokenizer training first or provide data/train.txt")
        if os.path.exists("data/train.txt"):
            tokenizer.train(["data/train.txt"])
        else:
            print("No training data found at data/train.txt. Exiting.")
            return

    # Update config vocab size based on tokenizer
    config.vocab_size = tokenizer.get_vocab_size()

    # Initialize model
    model = MiniGPT(config)
    
    # Optimizer
    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), device_type)

    iter_num = 0
    # Resume from checkpoint if it exists
    if os.path.exists("ckpt.pt"):
        print("Resuming from checkpoint...")
        checkpoint = torch.load("ckpt.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        iter_num = checkpoint['iter_num']
        config = checkpoint['config']
        print(f"Resuming from iteration {iter_num}")

    model.to(device)

    # Compile model if requested
    if config.compile and device_type == 'cuda':
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    # Dataloader
    if not os.path.exists("data/train.txt"):
        print("Waiting for data/train.txt...")
        return
        
    train_loader = get_dataloader("data/train.txt", tokenizer, config.batch_size, config.context_length)

    # LR Scheduler (Cosine Decay)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > config.lr_decay_iters:
            return config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)

    # Training Loop
    iter_num = 0
    t0 = time.time()
    
    model.train()
    print("Starting training...")
    
    for epoch in range(10): # Example: 10 epochs
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping
            if config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()
            
            # Logging
            if iter_num % 10 == 0:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, lr {lr:e}")
            
            iter_num += 1
            
            # Save checkpoint
            if iter_num % 500 == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iter_num': iter_num,
                }
                print(f"saving checkpoint to ckpt.pt")
                torch.save(checkpoint, 'ckpt.pt')

    # Final Save
    torch.save(model.state_dict(), "model.pt")
    print("Training finished. Model saved to model.pt")

if __name__ == "__main__":
    train()
