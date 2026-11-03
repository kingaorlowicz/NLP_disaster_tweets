# Natural Language Processing with Disaster Tweets
My solution to the Kaggle competition [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started).

**Current Kaggle score: 0.84952 (F1)**

## Overview

Twitter is used to report emergencies in real time, but not every tweet that mentions a disaster is about a real one. "The sky is on fire tonight" can be a metaphor just as easily as a wildfire report. The goal is to build a binary classifier that predicts whether a tweet is about a real disaster (`1`) or not (`0`).

- **Data:** ~7.6k labeled training tweets (`id`, `keyword`, `location`, `text`, `target`) and a test set to predict.
- **Metric:** F1 score.
- **Class balance:** ~57% non-disaster, ~43% disaster.

## Approach

The project explores classical ML baselines and advanced Deep Learning techniques using pre-trained Transformer architectures.

### 1. Baseline: TF-IDF + classical models (`code_baseline.ipynb`)

- Quick EDA: missing values (`keyword` ~1%, `location` ~33%), class balance, and keywords most strongly associated with real disasters.
- Text cleaning: lowercasing, removing URLs and @mentions, stripping punctuation, normalizing whitespace.
- The `keyword` column (with `%20` decoded) is prepended to the cleaned tweet to form a single input feature.
- Models, tuned with 5-fold `GridSearchCV` on F1:
  - TF-IDF + Logistic Regression: **validation F1 = 0.7773**
  - TF-IDF + linear SVC: **validation F1 = 0.7727**

### 2. Transformer: fine-tuned BERTweet + initial ensemble (`code_transformer.ipynb`)

- **Data cleaning:** removed duplicated tweets and tweets with conflicting labels (noisy labels).
- **Preprocessing:** minimal formatting expected by BERTweet (URLs -> `HTTPURL`, mentions -> `@USER`).
- **Model:** [`vinai/bertweet-base`](https://huggingface.co/vinai/bertweet-base) fine-tuned with Hugging Face `Trainer`.
- **Location Feature Engineering:** Attempted to prepend the `location` column to the text, using `geotext` to flag real geographical entities. Resulted in a slight score drop (**0.84462**), indicating the feature introduced more noise than signal.
- **Base Ensemble:** 3-fold stratified cross-validation. Averaging softmax probabilities of 3 BERTweet models yielded a strong baseline of **0.84554**.

### 3. Advanced Techniques: Dealing with Noisy Labels and Plateauing

To push beyond the ~0.845 limit, I experimented with techniques designed for datasets with mislabeled samples:

- **Hard-Voting Ensemble (BERTweet + RoBERTa) (`transformer_ensemble.ipynb`):** Trained a separate `roberta-base` model (**0.84216**) and combined it with the BERTweet using trust-based hard voting. This approach failed (**0.84370**) because the weaker model occasionally overruled correct predictions from the stronger one.
- **Pseudo-labeling (Semi-supervised learning) (`pseudo_labeling.ipynb`):** Used the best BERTweet model to predict the test set and appended the most confident predictions (>=95% confidence) to the training data. This lowered the score (**0.84247**) due to "confirmation bias" — the model confidently reinforced the dataset's original labeling errors.
- **Label Smoothing (`label_smoothing.ipynb`):** Applied to the core BERTweet model (`label_smoothing_factor=0.1`). Instead of forcing 100% confidence, the model aims for 90%, preventing it from aggressively overfitting to mislabeled tweets in the training set. This single change pushed the model to its mathematical peak on these data without leakage.

## Results

| Approach | Kaggle Public Score (F1) |
|---|---|
| TF-IDF + Logistic Regression | ~0.7773 (Local CV) |
| BERTweet + Location Feature Engineering | 0.84462 |
| BERTweet 3-fold ensemble (Base) | 0.84554 |
| Hard Voting Ensemble (BERTweet + RoBERTa) | 0.84370 |
| BERTweet with Pseudo-labeling | 0.84247 |
| **BERTweet with Label Smoothing (0.1)** | **0.84952** |

The single BERTweet model with label smoothing achieved **0.84952**, reaching the practical ceiling for this competition without exploiting data leakage (finding the original Figure Eight dataset).

## Repository structure
```
.
├── code_baseline.ipynb          # EDA, cleaning, TF-IDF + LR / SVC
├── code_transformer.ipynb       # Base BERTweet fine-tuning and cross-validation
├── transformer_ensemble.ipynb   # Hard-voting ensemble with RoBERTa
├── pseudo_labeling.ipynb        # Semi-supervised learning experiment
├── label_smoothing.ipynb        # Final BERTweet model with modified loss function
└── README.md                    # Project documentation
```
