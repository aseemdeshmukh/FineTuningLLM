def display_sample_errors(dataset, preds, num=5):
    wrong = [i for i, (p, t) in enumerate(zip(preds.predictions.argmax(axis=1), preds.label_ids)) if p != t]
    for idx in wrong[:num]:
        print(f"❌ Incorrectly Predicted: {dataset['test'][idx]['text'][:200]}...")
