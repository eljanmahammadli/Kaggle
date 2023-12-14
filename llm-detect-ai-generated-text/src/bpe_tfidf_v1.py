"""
1. **Tokenization and TF-IDF:**
   - Experiment with different tokenization techniques. You are currently using a Byte-Pair Encoding (BPE) tokenizer. You may try alternative tokenization methods, such as SentencePiece or WordPiece, to see if they have any impact on performance.

2. **TF-IDF Parameters:**
   - Fine-tune the parameters of your TF-IDF vectorizer. Adjusting parameters such as `ngram_range`, `sublinear_tf`, and `min_df` could potentially improve the representation of your text data.

3. **Feature Scaling:**
   - Consider applying feature scaling to your TF-IDF vectors. Some classifiers, like SGDClassifier, can benefit from scaled input features. You can use `MinMaxScaler` or `StandardScaler` from scikit-learn for this purpose.

4. **Model Parameters:**
   - Fine-tune the hyperparameters of your classifiers (`MultinomialNB`, `SGDClassifier`, `LGBMClassifier`, `CatBoostClassifier`). Experiment with different values and ranges to find the optimal settings.

5. **Ensemble Weights:**
   - Experiment with different weights for the models in your ensemble. The weights you assign to each model can significantly impact the ensemble's performance.

6. **Advanced Models:**
   - Consider trying more advanced models for text classification. Transformer-based models, such as BERT or RoBERTa, have shown great success in various NLP tasks. You can use pre-trained transformer models from the Hugging Face Transformers library.

7. **Regularization:**
   - Implement regularization techniques within your classifiers, especially for models like `SGDClassifier` and `LGBMClassifier`. This can help prevent overfitting.

8. **Cross-Validation:**
   - Implement cross-validation to obtain a more robust estimate of your model's performance. This can help ensure that your model generalizes well to unseen data.

11. **Data Preprocessing:**
    - Experiment with different text preprocessing techniques. For instance, you can try removing or stemming certain words, experimenting with different casing strategies, or handling rare words differently.
"""

# imports
import sys, gc
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

# import original and extra datasets
test = pd.read_csv("./data/test_essays.csv")
submission = pd.read_csv("./data/sample_submission.csv")
org_train = pd.read_csv("./data/train_essays.csv")
train = pd.read_csv("./data/train_v2_drcat_02.csv", sep=",")

train = train.drop_duplicates(subset=["text"])
train.reset_index(drop=True, inplace=True)
y_train = train["label"].values

LOWERCASE, VOCAB_SIZE = False, 30522

# Creating Byte-Pair Encoding tokenizer
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else []
)
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
dataset = Dataset.from_pandas(test[["text"]])


def train_corp_iter():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]


raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

tokenized_texts_test, tokenized_texts_train = [], []
for text in tqdm(test["text"].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))
for text in tqdm(train["text"].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))


def dummy(text):
    return text


tfidf_params = {
    "ngram_range": (3, 5),
    "lowercase": False,
    "sublinear_tf": True,
    "analyzer": "word",
    "tokenizer": dummy,
    "preprocessor": dummy,
    "token_pattern": None,
    "strip_accents": "unicode",
}

# learn TF-IDF vocabulary on test set
vectorizer = TfidfVectorizer(**tfidf_params)
vectorizer.fit(tokenized_texts_test)
vocab = vectorizer.vocabulary_
print(f"Test dataset TF-IDF vocabulary: {vocab}")

# fit TF-IDF on train set using only vocaulary learned from test set
vectorizer = TfidfVectorizer(**tfidf_params)
tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)
del vectorizer
gc.collect()

FLAG = True if len(test.text.values) <= 5 else False
if FLAG:
    submission.to_csv("./data/submission.csv", index=False)
else:
    clf = MultinomialNB(alpha=0.02)
    sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
    lgb_params = {
        "n_iter": 1500,
        "verbose": -1,
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05073909898961407,
        "colsample_bytree": 0.726023996436955,
        "colsample_bynode": 0.5803681307354022,
        "lambda_l1": 8.562963348932286,
        "lambda_l2": 4.893256185259296,
        "min_data_in_leaf": 115,
        "max_depth": 23,
        "max_bin": 898,
    }
    lgb = LGBMClassifier(**lgb_params)
    cat = CatBoostClassifier(
        iterations=1000,
        verbose=0,
        l2_leaf_reg=6.6591278779517808,
        learning_rate=0.005689066836106983,
        allow_const_label=True,
        loss_function="CrossEntropy",
    )
    weights = [0.07, 0.31, 0.31, 0.31]

    ensemble = VotingClassifier(
        estimators=[("mnb", clf), ("sgd", sgd_model), ("lgb", lgb), ("cat", cat)],
        weights=weights,
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(tf_train, y_train)
    gc.collect()
    final_preds = ensemble.predict_proba(tf_test)[:, 1]
    submission["generated"] = final_preds
    submission.to_csv("./data/submission.csv", index=False)
