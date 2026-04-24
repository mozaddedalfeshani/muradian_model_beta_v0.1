import os
import time
import math
import torch
from model import MuradianModel
from config import MuradianConfig
from tokenizer import BPETokenizer
from dataset import get_dataloader

def train():
    config = MuradianConfig()
    
    # Setup device
    device = config.device
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    # Set seed
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Initialize tokenizer
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    if os.path.exists("tokenizer.json"):
        tokenizer.load("tokenizer.json")
        config.vocab_size = tokenizer.get_vocab_size()
    else:
        print("Tokenizer not found. Please run train_tokenizer.py first.")
        return

    # Initialize model
    model = MuradianModel(config)
    
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
        print(f"Resuming from iteration {iter_num}")

    model.to(device)

    # Compile model if requested
    if config.compile and device_type == 'cuda':
        print("compiling the model...")
        model = torch.compile(model)

    # Data files
    data_files = ["data/train.txt", "data/train_bangla.txt"]
    existing_data = [f for f in data_files if os.path.exists(f)]
    if not existing_data:
        print("No training data found. Run prepare scripts first.")
        return
        
    print(f"Loading data from: {existing_data}")
    # Load all text and split
    full_text = ""
    for f in existing_data:
        with open(f, 'r', encoding='utf-8') as f_in:
            full_text += f_in.read() + "\n"
            
    # Simple train/val split (90% / 10%)
    n = len(full_text)
    train_text = full_text[:int(n*0.9)]
    val_text = full_text[int(n*0.9):]
    
    # We save these to temp files for the Dataloader to read
    # In a real scenario, InstructionDataset should handle text directly
    with open("data/train_split.txt", "w") as f: f.write(train_text)
    with open("data/val_split.txt", "w") as f: f.write(val_text)
    
    train_loader = get_dataloader("data/train_split.txt", tokenizer, config.batch_size, config.context_length)
    val_loader = get_dataloader("data/val_split.txt", tokenizer, config.batch_size, config.context_length)

    # LR Scheduler (Cosine Decay)
    def get_lr(it):
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        if it > config.lr_decay_iters:
            return config.min_lr
        decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = []
        # Check a few batches for validation
        count = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
            count += 1
            if count >= 20: break # Just estimate on 20 batches
        model.train()
        return sum(losses) / len(losses) if losses else 0

    @torch.no_grad()
    def generate_sample(prompt="User: কেমন আছ?\nAssistant:"):
        model.eval()
        idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        # Simple greedy generation
        for _ in range(30):
            idx_cond = idx[:, -config.context_length:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == tokenizer.tokenizer.token_to_id("<|endoftext|>"):
                break
        print(f"\n--- Sample ---\n{tokenizer.decode(idx[0].tolist())}\n--------------\n")
        model.train()

    # Training Loop
    t0 = time.time()
    print("Starting training...")
    
    for epoch in range(10):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()
            
            if iter_num % 100 == 0:
                val_loss = estimate_loss()
                t1 = time.time()
                print(f"iter {iter_num}: loss {loss.item():.4f}, val_loss {val_loss:.4f}, time {(t1-t0)*1000/100:.2f}ms/it, lr {lr:e}")
                t0 = t1
                generate_sample()
            
            iter_num += 1
            
            if iter_num % 1000 == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iter_num': iter_num,
                }
                torch.save(checkpoint, 'ckpt.pt')

    torch.save(model.state_dict(), "model.pt")
    print("Training finished.")

if __name__ == "__main__":
    train()

if __name__ == "__main__":
    train()
