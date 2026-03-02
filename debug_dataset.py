from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()

def check_dataset():
    hf_token = os.environ.get("HF_TOKEN")
    print(f"Loading dataset capleaf/viVoice...")
    dataset = load_dataset("capleaf/viVoice", split="train", streaming=True, token=hf_token)
    
    for i, example in enumerate(dataset):
        if i >= 5:
            break
        print(f"Index {i}: {example['text']}")

if __name__ == "__main__":
    check_dataset()
