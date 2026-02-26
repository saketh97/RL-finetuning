from transformers import AutoTokenizer
from dataset_loader import load_and_prepare_dataset
import config

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

dataset = load_and_prepare_dataset(tokenizer, max_samples=5)

for i in range(3):
    print("="*80)
    print(dataset[i]["text"])