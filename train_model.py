"""Train and evaluate the SMS spam classifier."""

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "spam.csv"
MODEL_DIR = PROJECT_DIR / "model"
MODEL_PATH = MODEL_DIR / "spam_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


def clean_message(message: str) -> str:
    """Normalize case and whitespace before vectorizing a message."""
    return re.sub(r"\s+", " ", str(message).lower()).strip()


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.Series, pd.Series]:
    """Read common spam-dataset column names and return clean texts and labels."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Add data/spam.csv and run this script again."
        )

    data = pd.read_csv(path, encoding="utf-8-sig")
    columns = {str(column).strip().lower(): column for column in data.columns}
    label_column = next(
        (columns[name] for name in ("label", "category", "v1", "class", "target") if name in columns),
        None,
    )
    message_column = next(
        (columns[name] for name in ("message", "text", "v2", "sms", "email") if name in columns),
        None,
    )
    if label_column is None or message_column is None:
        raise ValueError(
            "The CSV must contain a label column (label/category/v1) and a "
            "message column (message/text/v2)."
        )

    data = data[[label_column, message_column]].copy()
    data.columns = ["label", "message"]
    data = data.dropna(subset=["label"])
    data["message"] = data["message"].fillna("").map(clean_message)
    data = data[data["message"] != ""]

    label_map = {
        "spam": 1,
        "junk": 1,
        "1": 1,
        "ham": 0,
        "not spam": 0,
        "legitimate": 0,
        "0": 0,
    }
    data["label"] = data["label"].astype(str).str.strip().str.lower().map(label_map)
    data = data.dropna(subset=["label"])
    if data.empty or set(data["label"].astype(int).unique()) != {0, 1}:
        raise ValueError("The dataset must contain both ham and spam examples.")

    return data["message"], data["label"].astype(int)


def train_and_evaluate() -> None:
    messages, labels = load_dataset()
    class_ids, class_counts = np.unique(labels.to_numpy(), return_counts=True)
    print("Class distribution:")
    for class_id, count in zip(class_ids, class_counts):
        class_name = "SPAM" if class_id == 1 else "HAM"
        print(f"  {class_name}: {count:,}")

    if labels.value_counts().min() < 2:
        raise ValueError("Each class needs at least two rows for a stratified train/test split.")

    x_train, x_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
    )
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_tfidf, y_train)
    predictions = model.predict(x_test_tfidf)

    print(f"Dataset rows used: {len(messages):,}")
    print(f"Training rows: {len(x_train):,} | Testing rows: {len(x_test):,}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    print("Confusion matrix (rows=true, columns=predicted; ham, spam):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved model: {MODEL_PATH.relative_to(PROJECT_DIR)}")
    print(f"Saved vectorizer: {VECTORIZER_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    try:
        train_and_evaluate()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"Training could not finish: {error}") from error