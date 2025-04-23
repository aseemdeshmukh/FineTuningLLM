from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from data_prep import load_and_tokenize

# ✅ STEP 1: Load tokenized dataset
print("🛠 Loading and tokenizing dataset...")
dataset = load_and_tokenize()
print("✅ Dataset loaded.")

# ✅ STEP 2: Load model and tokenizer
print("🛠 Loading BERT model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("✅ Model and tokenizer ready.")

# ✅ STEP 3: Define training arguments
print("🛠 Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",                  # Where to save the model
    num_train_epochs=3,                      # Number of epochs
    per_device_train_batch_size=8,           # Training batch size
    per_device_eval_batch_size=8,            # Evaluation batch size
    evaluation_strategy="epoch",             # Run eval each epoch
    save_strategy="epoch",                   # Save checkpoint after each epoch
    logging_dir="./logs",                    # Where to log
    load_best_model_at_end=True,             # Restore best model after training
    logging_steps=50,                        # Log every 50 steps
    # Uncomment next line to debug faster:
    max_steps=10,                          # ⛔️ Only 10 steps for testing — REMOVE in final version!
)
print("✅ Training arguments set.")

# ✅ STEP 4: Initialize the Trainer
print("🛠 Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,  # Important: this must be the actual tokenizer object
)
print("✅ Trainer ready.")

# ✅ STEP 5: Start training
print("🚀 Starting training...")
trainer.train()
print("✅ Training complete!")

# ✅ Done!
print("📦 Model saved at: ./results")
