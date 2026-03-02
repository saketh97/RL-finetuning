from transformers import AutoTokenizer
from training.dataset_loader import load_and_prepare_dataset
import training.config as config
from langfuse import Langfuse
from dotenv import load_dotenv
import os
import torch
from transformers import BitsAndBytesConfig,AutoModelForCausalLM
from langfuse import get_client

load_dotenv()
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL=os.getenv("LANGFUSE_BASE_URL")

langfuse = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY,host=LANGFUSE_BASE_URL)

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

def run_model(prompt, max_new_tokens=32, temperature=0.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

dataset=langfuse.get_dataset("flare-cfa")
def run_task(*, item,**kwargs):
    messages=item.input
    response = run_model(messages)
    return response

result = dataset.run_experiment(
    name="base_model_validation",
    descrption="evaluating the base tiny llama model",
    task=run_task
)

get_client().flush()
