# SMS/Email Spam Detector

A beginner-friendly text-classification project that predicts whether a message is **SPAM** or **NOT SPAM (HAM)**. It uses TF-IDF to represent message text as numbers and a Multinomial Naive Bayes classifier to make predictions. A Streamlit page provides a simple interface for trying messages.

## Objectives

- Prepare and inspect a real-world labeled SMS dataset.
- Train and evaluate a text classifier using a repeatable train/test split.
- Save the trained classifier and TF-IDF vectorizer for later use.
- Predict a message from a local web application without retraining on each page load.

## Features

- Multinomial Naive Bayes text classification.
- TF-IDF features using single words and two-word phrases.
- Accuracy, precision, recall, F1-score, and confusion-matrix evaluation.
- Spam likelihood estimate for each entered message.
- Friendly messages for empty input or missing model files.

## Technologies

Python, Pandas, NumPy, scikit-learn, Streamlit, and Joblib. The project uses no database or external service at runtime.

## Algorithm

The main classifier is **Multinomial Naive Bayes**. It learns how often words and phrases occur in spam and ham messages, then uses Bayes' theorem to estimate which class best fits a new message. The probabilities use a simplifying assumption that features contribute independently. Although words are not truly independent, this fast, simple model is a strong baseline for text classification.

## Dataset

`data/spam.csv` is a CSV-formatted copy of the public [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). It contains 5,574 labeled SMS messages, originally distributed as tab-separated `ham`/`spam` and message fields. The supplied CSV uses the column names `label` and `message`. Ham means legitimate; spam means unwanted or fraudulent. The source data is SMS, so email predictions are an educational demonstration and are not guaranteed to generalize to email.

## How the System Works

1. Read the labeled data, discard rows without a label, fill missing message text, and normalize case and whitespace.
2. Convert `ham` to `0` and `spam` to `1`.
3. Split the examples into training (80%) and testing (20%) sets, retaining the class balance.
4. Fit TF-IDF on the training messages only, then use it to represent both sets as numeric word/phrase features.
5. Train Multinomial Naive Bayes on the training features and labels.
6. Evaluate predictions on the held-out testing set and save the model and vectorizer under `model/`.
7. The Streamlit app loads those saved files, transforms a user's message with the same vectorizer, and displays the predicted class and spam likelihood.

**Input Text → TF-IDF → Machine Learning Model → Spam / Not Spam**

## Project Structure

```text
sms-email-spam-detector/
├── .streamlit/
│   └── config.toml
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── spam.csv
└── model/
    ├── spam_model.pkl
    └── tfidf_vectorizer.pkl
```

The two files in `model/` are created by the training script.

## Installation (Windows)

Open PowerShell in the project folder. Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use Command Prompt and run `.venv\Scripts\activate.bat`, then continue with the same pip commands.

## Train the Model

```powershell
python train_model.py
```

The script prints the evaluation metrics and confusion matrix, then creates `model/spam_model.pkl` and `model/tfidf_vectorizer.pkl`. It does not run automatically when Streamlit starts.

## Run the Application

```powershell
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Paste a message into the text area and select **Detect Spam**. If model files are absent, train the model first.

## Example Inputs and Outputs

- `Congratulations! You have won a free prize. Call now to claim.` → likely **SPAM**.
- `Hi, are we still meeting for lunch today?` → likely **NOT SPAM (HAM)**.

These are examples, not guaranteed outcomes. The percentage shown is the model's estimated spam probability, not a guarantee that a message is safe or malicious.

## Viva-Friendly Explanation

### What is spam?

Spam is an unwanted message, often advertising something, asking for personal information, or trying to trick its reader. Ham is a normal, legitimate message.

### What is Machine Learning?

Machine Learning is a way for a computer to learn patterns from example data and use those patterns to make predictions on new data.

### What is text classification?

Text classification assigns a piece of text to one or more categories. Here the categories are spam and ham.

### What is TF-IDF, and why use it?

TF-IDF means **Term Frequency–Inverse Document Frequency**. It gives a word a larger weight when it appears often in one message but is less common across all messages. Common words that appear everywhere receive less weight. This turns text into useful numeric features while reducing the influence of very common words.

### What is Naive Bayes?

Naive Bayes is a family of classifiers based on Bayes' theorem. It estimates the probability of each class given the observed features, using a simplifying assumption that the features are independent.

### Why Multinomial Naive Bayes here?

Multinomial Naive Bayes is fast, easy to explain, and designed for non-negative feature values such as word counts or TF-IDF weights. It is a practical baseline for spam filtering and works well with sparse text features.

### What are training and testing datasets?

The training set is the portion of labeled examples used to learn the model. The testing set is held aside during training and used afterward to estimate how well the model predicts messages it has not seen. This project uses an 80/20 split.

### What are accuracy, precision, recall, and F1-score?

- **Accuracy:** the fraction of all predictions that are correct.
- **Precision:** among messages predicted as spam, the fraction that really are spam.
- **Recall:** among all actual spam messages, the fraction the model finds.
- **F1-score:** a single score combining precision and recall using their harmonic mean; it is useful when both kinds of error matter.

### What is a confusion matrix?

A confusion matrix counts correct and incorrect predictions by actual and predicted class. The printed matrix uses rows for the actual class and columns for the predicted class, ordered ham then spam. It shows ham correctly identified, ham incorrectly flagged, spam missed, and spam correctly caught.

## Future Improvements

- Evaluate on a separate, newer dataset, especially real email data.
- Compare Naive Bayes with Logistic Regression and tune thresholds for the cost of missed spam versus false alarms.
- Add clearer message preprocessing and a larger multilingual dataset.
- Add automated tests and a model-version/data-quality report.