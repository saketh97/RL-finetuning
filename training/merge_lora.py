import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import config
BASE_MODEL = config.MODEL_NAME
LORA_PATH = config.OUTPUT_DIR
OUTPUT_PATH = "../artifacts/merged_model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    low_cpu_mem_usage=False
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("Merging LoRA weights...")
merged_model = model.merge_and_unload()

merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("Done! Merged model saved to:", OUTPUT_PATH)