from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize(tokenizer_name="bert-base-uncased", max_length=512):
    # Load the IMDB dataset
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Define the tokenization function
    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    # Split training set into 90% train, 10% validation
    train_valid = dataset["train"].train_test_split(test_size=0.1)
    test = dataset["test"]  

    # Tokenize each split
    tokenized_train = train_valid["train"].map(tokenize_fn, batched=True)
    tokenized_valid = train_valid["test"].map(tokenize_fn, batched=True)
    tokenized_test = test.map(tokenize_fn, batched=True)

    # Format all sets for PyTorch
    for ds in [tokenized_train, tokenized_valid, tokenized_test]:
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Return as a dictionary
    return {
        "train": tokenized_train,
        "validation": tokenized_valid,
        "test": tokenized_test
    }
