from datasets import load_dataset
import os
import json
import re

def clean_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("Loading dataset Mahadih534/alpaca_chat_bangla...")
    dataset = load_dataset("Mahadih534/alpaca_chat_bangla")
    
    output_text_file = "data/train_bangla.txt"
    os.makedirs("data", exist_ok=True)
    
    # We use <|endoftext|> as the separator between conversations
    EOS_TOKEN = "<|endoftext|>"
    
    print(f"Converting and saving to {output_text_file} with cleaning and proper formatting...")
    
    count = 0
    with open(output_text_file, "w", encoding="utf-8") as f:
        for entry in dataset["train"]:
            conv_str = entry["conversations"]
            
            # Simple regex to extract content and role
            # Given the format: [{'content': '...', 'role': 'user'}, {'content': '...', 'role': 'assistant'}]
            parts = re.findall(r"'content':\s*'(.*?)',\s*'role':\s*'(.*?)'", conv_str, re.DOTALL)
            
            if not parts:
                continue
                
            formatted_conv = ""
            for content, role in parts:
                role_name = "User" if role == "user" else "Assistant"
                # Decode unicode
                try:
                    content = content.encode().decode('unicode_escape')
                except:
                    pass
                
                content = clean_text(content)
                formatted_conv += f"{role_name}: {content}\n"
            
            # Add EOS token after each conversation
            f.write(formatted_conv + EOS_TOKEN + "\n\n")
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} conversations...")

    print(f"Done! Processed {count} conversations.")

if __name__ == "__main__":
    main()
