from transformers import AutoTokenizer
from dataset_loader import load_and_prepare_dataset
import config

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

dataset = load_and_prepare_dataset(tokenizer, max_samples=50000)


import numpy as np

lengths = np.array([len(x["text"]) for x in dataset])

print("Text length stats (characters):")
print("Min     :", lengths.min())
print("Median  :", int(np.median(lengths)))
print("Mean    :", int(lengths.mean()))
print("P90     :", int(np.percentile(lengths, 90)))
print("P95     :", int(np.percentile(lengths, 95)))
print("Max     :", lengths.max())

#for i in range(3):
   # print("="*80)
  #  print(dataset[i]["text"])