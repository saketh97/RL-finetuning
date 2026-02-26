from datasets import load_dataset


def load_and_prepare_dataset(tokenizer, max_samples=None):

    dataset = load_dataset(
        "FinLang/investopedia-instruction-tuning-dataset",
        split="train"
    )

    if max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    def format_prompt(example):
        prompt = (
            "<|system|>You are a professional financial analyst and educator. "
            "Provide accurate, clear, and concise financial explanations.\n"
            f"<|user|>{example['Question']}\n"
            f"<|assistant|>{example['Answer']}"
        )
        return {"text": prompt}

    dataset = dataset.map(
        format_prompt,
        remove_columns=dataset.column_names
    )

    return dataset