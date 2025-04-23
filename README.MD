# IMDB Sentiment Fine-Tuning - Hugging Face Transformers

This project fine-tunes a pre-trained BERT model (`bert-base-uncased`) on the IMDB movie review dataset for binary sentiment classification (positive/negative).

## 📦 File Structure
- `data_prep.py` - Loads and tokenizes IMDB dataset
- `train.py` - Fine-tunes the model
- `evaluate.py` - Evaluates model performance
- `inference.py` - Loads fine-tuned model for inference
- `utils.py` - Optional helper functions
- `config.json` - Stores hyperparameters
- `requirements.txt` - Required packages

##  How to Run

```bash
pip install -r requirements.txt
