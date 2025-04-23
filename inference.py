from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("results")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def predict_sentiment(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    label = torch.argmax(output.logits, dim=1).item()
    return "Positive 😊" if label == 1 else "Negative 😞"

# Example use
if __name__ == "__main__":
    text = input("Enter review: ")
    print("Sentiment:", predict_sentiment(text))
