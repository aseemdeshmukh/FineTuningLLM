from train import model, trainer, dataset
from sklearn.metrics import classification_report

# Run prediction
print("🚀 Running model on test set...")
preds = trainer.predict(dataset["test"])
y_pred = preds.predictions.argmax(axis=1)
y_true = preds.label_ids

# Print classification report
print("✅ Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))
