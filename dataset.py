import torch
from torch.utils.data import Dataset
import os

class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, context_length=512):
        self.tokenizer = tokenizer
        self.context_length = context_length
        
        # Load and tokenize the entire dataset
        # For larger datasets, this should be done lazily or pre-tokenized
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # Simple instruction format parsing if needed, 
        # but usually we just tokenize the whole thing with special separators
        # User: ... Assistant: ...
        tokens = self.tokenizer.encode(raw_text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Pad if tokens are fewer than context_length + 1
        if len(self.tokens) < self.context_length + 1:
            padding_needed = (self.context_length + 1) - len(self.tokens)
            padding = torch.full((padding_needed,), self.tokenizer.tokenizer.token_to_id("[PAD]"), dtype=torch.long)
            self.tokens = torch.cat([self.tokens, padding])
            
        print(f"Dataset loaded: {len(self.tokens)} tokens")

    def __len__(self):
        # Ensure at least 1 sample
        return max(1, len(self.tokens) - self.context_length)

    def __getitem__(self, idx):
        # Grab a chunk of context_length + 1 (for labels)
        chunk = self.tokens[idx : idx + self.context_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_dataloader(data_path, tokenizer, batch_size, context_length, shuffle=True):
    dataset = InstructionDataset(data_path, tokenizer, context_length)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        pin_memory=True
    )
