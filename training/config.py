MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "../artifacts/finetuned_model"

MAX_SEQ_LENGTH = 1024

BATCH_SIZE = 1
GRAD_ACCUM = 4
EPOCHS = 1
LR = 2e-4

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05