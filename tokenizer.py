import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

class BPETokenizer:
    def __init__(self, vocab_size=8192):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "<|endoftext|>" # Using same for BOS/EOS for simplicity
        self.eos_token = "<|endoftext|>"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token]

    def train(self, files, save_path="tokenizer.json"):
        """Train a BPE tokenizer on the provided files."""
        tokenizer = Tokenizer(models.BPE(unk_token=self.unk_token))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        
        tokenizer.train(files, trainer)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()
        
        self.tokenizer = tokenizer
        self.tokenizer.save(save_path)
        print(f"Tokenizer trained and saved to {save_path}")

    def load(self, path="tokenizer.json"):
        """Load a saved tokenizer."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found at {path}")
        self.tokenizer = Tokenizer.from_file(path)
        self.tokenizer.decoder = decoders.ByteLevel()

    def encode(self, text):
        """Encode text to token IDs."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded or trained.")
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        """Decode token IDs back to text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded or trained.")
        return self.tokenizer.decode(ids)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def __len__(self):
        return self.get_vocab_size()

if __name__ == "__main__":
    # Quick test/demo
    import tempfile
    
    # Create dummy data
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("User: Hello!\nAssistant: Hi there! How can I help you today?\n")
        f.write("User: Tell me a joke.\nAssistant: Why did the chicken cross the road?\n")
        dummy_file = f.name
    
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train([dummy_file])
    
    text = "User: Hello!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    
    os.unlink(dummy_file)
