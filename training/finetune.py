import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import config
from dataset_loader import load_and_prepare_dataset
from transformers import BitsAndBytesConfig


def tokenize(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=config.MAX_SEQ_LENGTH
    )


def main():
    # data loading  -> instruction style finetuning format -> batched tokenization with max_length truncking
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_and_prepare_dataset(tokenizer, max_samples=50000)  # debug run

    tokenized_dataset = dataset.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    
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

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)


if __name__ == "__main__":
    main()