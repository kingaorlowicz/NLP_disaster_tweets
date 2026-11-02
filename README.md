# Natural Language Processing with Disaster Tweets
My solution to the Kaggle competition [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started).

**Current Kaggle score: 0.84554 (F1)**

## Overview

Twitter is used to report emergencies in real time, but not every tweet that mentions a disaster is about a real one. "The sky is on fire tonight" can be a metaphor just as easily as a wildfire report. The goal is to build a binary classifier that predicts whether a tweet is about a real disaster (`1`) or not (`0`).

- **Data:** ~7.6k labeled training tweets (`id`, `keyword`, `location`, `text`, `target`) and a test set to predict.
- **Metric:** F1 score.
- **Class balance:** ~57% non-disaster, ~43% disaster.

## Approach

The project is split into two notebooks.

### 1. Baseline: TF-IDF + classical models (`code_baseline.ipynb`)

- Quick EDA: missing values (`keyword` ~1%, `location` ~33%), class balance, and keywords most strongly associated with real disasters.
- Text cleaning: lowercasing, removing URLs and @mentions, stripping punctuation, normalizing whitespace.
- The `keyword` column (with `%20` decoded) is prepended to the cleaned tweet to form a single input feature.
- Models, tuned with 5-fold `GridSearchCV` on F1:
  - TF-IDF + Logistic Regression: **validation F1 = 0.7773**
  - TF-IDF + linear SVC: **validation F1 = 0.7727**

### 2. Transformer: fine-tuned BERTweet + ensemble (`code_transformer.ipynb`)

- **Data cleaning:** removed duplicated tweets and tweets with conflicting labels, which are label noise.
- **Preprocessing:** minimal, in the format BERTweet expects. URLs become `HTTPURL` and mentions become `@USER`. Punctuation and casing are kept because the model uses them.
- **Model:** [`vinai/bertweet-base`](https://huggingface.co/vinai/bertweet-base), a RoBERTa-style model pretrained on English tweets, fine-tuned with Hugging Face `Trainer` (3 epochs, lr 2e-5, batch size 16, max length 128).
- **Error analysis:** inspected false positives and false negatives on the validation split.
- **Ensemble:** 3-fold stratified cross-validation. A separate BERTweet is fine-tuned on each fold, and the softmax probabilities of the three models on the test set are averaged to produce the final prediction.

## Results

| Approach | Score |
|---|---|
| TF-IDF + Logistic Regression | 0.7773 (validation F1) |
| TF-IDF + SVC | 0.7727 (validation F1) |
| **BERTweet 3-fold ensemble** | **0.84554 (Kaggle public leaderboard)** |

The BERTweet ensemble is currently the best-performing approach.

## Repository structure

```
.
├── code_baseline.ipynb      # EDA, cleaning, TF-IDF + LR / SVC
├── code_transformer.ipynb   # BERTweet fine-tuning + 3-fold ensemble
└── README.md
```


## Possible next steps
- Tune the learning rate, number of epochs, and max sequence length.
- Use the `keyword` and `location` columns as additional input to the transformer.
