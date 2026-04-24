from tokenizer import BPETokenizer
import os

def main():
    # Step 4: Scale slightly - increase vocab size to 16k
    vocab_size = 16384
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    
    # We'll train on the new Bangla text file and the original personal data
    data_files = ["data/train_bangla.txt", "data/train.txt"]
    existing_files = [f for f in data_files if os.path.exists(f)]
    
    if not existing_files:
        print("No training data found. Please run prepare scripts first.")
        return
        
    print(f"Training tokenizer on: {existing_files} with vocab size {vocab_size}...")
    tokenizer.train(existing_files, save_path="tokenizer.json")
    print("Tokenizer training complete!")

if __name__ == "__main__":
    main()
