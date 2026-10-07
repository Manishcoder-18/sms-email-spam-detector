# Project Report: SMS/Email Spam Detector

## 1. Overview
This project is a beginner-friendly machine learning application for detecting whether a text message is likely spam or ham (not spam). It combines text preprocessing, TF-IDF feature extraction, and a Multinomial Naive Bayes classifier to classify incoming messages.

The app is implemented using Python and deployed as a local Streamlit web interface. It allows a user to paste a message and instantly receive a spam probability and prediction.

## 2. Project Goal
The main aim of the project is to demonstrate how a real-world text classification model can:
- learn from labeled message data,
- convert raw text into numerical features,
- train a machine learning classifier,
- save the trained model for reuse,
- and provide a simple user interface for predictions.

## 3. Core Functionality
The system performs the following tasks:
1. Reads a dataset of labeled SMS messages.
2. Cleans and normalizes the text data.
3. Converts text into TF-IDF features.
4. Splits the data into training and testing sets.
5. Trains a Multinomial Naive Bayes classifier.
6. Evaluates the model using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Saves the trained model and vectorizer.
8. Loads them in a Streamlit app for live predictions.

## 4. Technologies Used
- Python
- Pandas
- NumPy
- scikit-learn
- Streamlit
- Joblib

## 5. Project Files
- `app.py` – Streamlit web app for interacting with the model.
- `train_model.py` – Script to train and evaluate the classifier.
- `data/spam.csv` – Dataset used for training.
- `model/` – Directory where trained model files are stored.
- `requirements.txt` – Python dependencies.
- `README.md` – User documentation.

## 6. Dataset
The project uses the public SMS Spam Collection dataset, which contains labeled SMS messages categorized as either spam or ham. The data is stored in `data/spam.csv`.

The script loads the dataset, validates required columns, drops empty or invalid entries, and maps labels into binary values:
- ham = 0
- spam = 1

## 7. Machine Learning Approach
### Feature Extraction
The text is converted into numeric features using TF-IDF (Term Frequency-Inverse Document Frequency). This method gives more weight to important words while reducing the impact of very common words.

### Model
The classifier used is Multinomial Naive Bayes, which works well for text classification tasks and is simple, fast, and effective for this type of problem.

### Training and Evaluation
The dataset is split into training and testing subsets using an 80/20 stratified split. The model is trained on the training set and then evaluated on unseen test data. The evaluation includes:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## 8. Application Workflow
The application workflow is as follows:
1. User enters a message in the Streamlit text area.
2. The text is transformed using the saved TF-IDF vectorizer.
3. The trained model predicts whether the message is spam or ham.
4. The app displays:
   - the prediction,
   - the spam likelihood percentage,
   - and a warning/error if the message is empty or the model is missing.

## 9. System Behavior
The project is designed to avoid retraining the model on every page load. Instead, the trained model and vectorizer are saved to disk and reused when the app runs.

If the model files do not exist, the app shows a helpful message telling the user to run the training script first.

## 10. Strengths of the Project
- Simple and easy to understand for beginners.
- Clean separation between training and deployment logic.
- Uses a realistic text dataset.
- Provides a practical demonstration of machine learning in a web app.
- Easy to run locally with minimal setup.

## 11. Limitations
- The model is trained on SMS data, not a full email dataset.
- Performance may vary on very short or unusual messages.
- A Naive Bayes baseline is simple but not always the strongest classifier.
- The project is not designed for production-scale spam filtering.

## 12. Suggested Improvements
- Test on a larger and more recent spam dataset.
- Compare Naive Bayes with logistic regression or SVM.
- Tune model parameters for better performance.
- Add unit tests for training and prediction logic.
- Improve preprocessing and message cleaning.
- Expand the app with message history and model versioning.

## 13. Conclusion
This project is a solid introductory machine learning application that demonstrates text classification, basic model evaluation, and deployment through a user-friendly Streamlit interface. It is especially useful for learning how spam detection works using real text data and classic NLP techniques.

Overall, the project successfully combines educational value with an easy-to-use practical example of applying machine learning to a real-world problem.
