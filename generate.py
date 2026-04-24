import os
import torch
import torch.nn.functional as F
from model import MuradianModel
from config import MuradianConfig
from tokenizer import BPETokenizer

class Generator:
    def __init__(self, model_path, tokenizer_path="tokenizer.json", device='cuda'):
        self.device = device
        
        # Load tokenizer
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)
        
        # Load config and model
        # For simplicity, we assume we can reconstruct config or it's saved
        # Here we use default config and override vocab_size
        config = MuradianConfig()
        config.vocab_size = self.tokenizer.get_vocab_size()
        
        self.model = MuradianModel(config)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle potential torch.compile prefix
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text starting from prompt.
        """
        idx = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at context_length
            idx_cond = idx if idx.size(1) <= self.model.config.context_length else idx[:, -self.model.config.context_length:]
            
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.model(idx_cond)
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # optionally apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop if we hit the end of text token
            if idx_next.item() == self.tokenizer.tokenizer.token_to_id("<|endoftext|>"):
                break

        return self.tokenizer.decode(idx[0].tolist())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="User: Hello!\nAssistant:")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    # Determine model path
    if args.model is None:
        if os.path.exists("model.pt"):
            args.model = "model.pt"
        elif os.path.exists("ckpt.pt"):
            args.model = "ckpt.pt"
        else:
            print("Neither model.pt nor ckpt.pt found. Train the model first.")
            exit(1)
            
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        generator = Generator(args.model, device=device)
        output = generator.generate(
            args.prompt, 
            max_new_tokens=args.tokens, 
            temperature=args.temp, 
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(f"\n--- Generated ---\n{output}\n-----------------")
    except Exception as e:
        print(f"Error during generation: {e}")
        print("Make sure you have trained the model and 'model.pt' exists.")
