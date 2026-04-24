from generate import Generator
import torch
import os

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "model.pt"
    if not os.path.exists(model_path):
        if os.path.exists("ckpt.pt"):
            model_path = "ckpt.pt"
        else:
            print("No model.pt or ckpt.pt found. Please train the model first.")
            return

    print(f"Loading model from {model_path}...")
    try:
        gen = Generator(model_path, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This usually happens if the model architecture in config.py doesn't match the saved weights.")
        return

    questions = [
        "কেমন আছ?",
        "তোমার নাম কি?",
        "মুরাদ কে?",
        "একটি কৌতুক বল।",
        "ঢাকা সম্পর্কে কিছু বল।"
    ]

    print("\n--- Running Tests ---\n")
    for q in questions:
        prompt = f"User: {q}\nAssistant:"
        print(f"Q: {q}")
        response = gen.generate(prompt, max_new_tokens=50, temperature=0.7)
        # Extract assistant response
        if "Assistant:" in response:
            res = response.split("Assistant:")[1].strip()
        else:
            res = response
        print(f"A: {res}\n")

if __name__ == "__main__":
    main()
