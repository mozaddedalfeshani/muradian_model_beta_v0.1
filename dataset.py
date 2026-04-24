import torch
from torch.utils.data import Dataset
import os
import json
import re

class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, context_length=512):
        self.tokenizer = tokenizer
        self.context_length = context_length
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} not found.")
            
        with open(data_path, 'r', encoding='utf-8') as f:
            all_text = f.read()
        
        if not all_text:
            all_text = "User: Hello\nAssistant: Hi there!\n"
            
        # Tokenize the entire dataset
        tokens = self.tokenizer.encode(all_text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Pad if tokens are fewer than context_length + 1
        if len(self.tokens) < self.context_length + 1:
            padding_needed = (self.context_length + 1) - len(self.tokens)
            pad_id = self.tokenizer.tokenizer.token_to_id("[PAD]")
            if pad_id is None: pad_id = 0
            padding = torch.full((padding_needed,), pad_id, dtype=torch.long)
            self.tokens = torch.cat([self.tokens, padding])
            
        print(f"Dataset {data_path} loaded: {len(self.tokens)} tokens")

    def __len__(self):
        return max(1, len(self.tokens) - self.context_length)

    def __getitem__(self, idx):
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
