# Tweet Sentiment Analysis (Ensemble Model)

A binary sentiment classifier for tweets (positive vs. negative) built as an ensemble of three models: **Naive Bayes**, **Logistic Regression**, and **XGBoost**. Predictions from all three are averaged to produce the final classification.

## Dataset

This project uses the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) from Kaggle — 1.6 million labeled tweets.

| Column  | Description |
|---------|-------------|
| `target`| Sentiment label (`0` = negative, `2` = neutral, `4` = positive) |
| `ids`   | Tweet ID |
| `date`  | Timestamp of the tweet |
| `flag`  | Query used (or `NO_QUERY`) |
| `user`  | Username of the tweeter |
| `text`  | Tweet content |

Download the CSV and place it at `Downloads/sentiment140.csv` relative to the script, or update the path in `code.py`.

## How It Works

1. **Load & clean the data** — keep only `text` and `target`, drop neutral tweets, and remap labels to binary (`0` = negative, `1` = positive).
2. **Split** into train/test sets (80/20).
3. **Vectorize** tweet text with `CountVectorizer` (bag-of-words).
4. **Train three models** on the vectorized text:
   - Multinomial Naive Bayes
   - Logistic Regression
   - XGBoost Classifier
5. **Ensemble** — average the predicted probabilities from all three models and classify as positive if the averaged probability exceeds 0.5.
6. **Evaluate** using accuracy, a classification report, and a confusion matrix.
7. **Interactive prediction** — the script prompts for a tweet, preprocesses it (lowercasing, URL removal, punctuation stripping), and prints the ensemble's predicted sentiment.

## Requirements

```
pandas
scikit-learn
xgboost
```

Install with:

```bash
pip install pandas scikit-learn xgboost
```

## Usage

```bash
python code.py
```

The script will train all three models, print evaluation metrics on the test set, then prompt you to enter a tweet for a live sentiment prediction.

## Notes / Possible Improvements

- `CountVectorizer` could be swapped for `TfidfVectorizer` for potentially better results.
- Model performance depends on preprocessing choices — consider stemming/lemmatization or removing stopwords.
- Training XGBoost on the full 1.6M-row dataset can be slow; consider a subsample for experimentation.
- Results (accuracy, precision/recall) will vary by run — check the printed classification report for current numbers rather than relying on a fixed figure.
