import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from dataset_loader import load_and_prepare_dataset
import config

def main():
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token=tokenizer.eos_token
    
    dataset = load_and_prepare_dataset(tokenizer, max_samples=2000)
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        load_in_4bit=True,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        num_train_epochs=config.EPOCHS,
        learning_rate=config.LR,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH
    )
    
    trainer.train()

    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)


if __name__ == "__main__":
    main()