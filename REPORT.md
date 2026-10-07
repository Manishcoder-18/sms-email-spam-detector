# Design and Development of a Spam Detection System for SMS and Email Messages Using TF-IDF and Multinomial Naive Bayes

## Abstract

This project presents the design and implementation of a machine learning-based spam detection system for text messages. The system uses TF-IDF to transform textual data into numerical features and applies a Multinomial Naive Bayes classifier to classify messages as either spam or ham. A labelled SMS dataset is used for training and evaluation, and the final model is deployed in a Streamlit web application to provide a user-friendly interface for prediction. The project demonstrates that a simple but effective text classification pipeline can be used to solve a real-world problem in digital communication. The results show that the combination of TF-IDF and Naive Bayes is efficient, easy to implement, and well suited for short-text classification tasks. This makes the project suitable for academic study, demonstration, and lightweight practical deployment.

## 1. Introduction

Digital communication has become an essential part of modern life, but it has also increased the spread of unwanted messages, including spam and fraudulent content. Spam messages are often disruptive and may contain deceptive offers, phishing attempts, or irrelevant promotional content. Therefore, there is a need for effective automated systems that can identify and filter such messages.

Traditional spam filtering techniques, such as keyword matching, are limited because spam messages change constantly and may use different wording to avoid detection. Machine learning provides a more flexible solution by learning from examples. Text classification models can identify the features that distinguish spam from legitimate messages and apply those patterns to new inputs.

This project focuses on the development of a spam detection system for SMS and email-like messages. The model uses TF-IDF feature extraction and a Multinomial Naive Bayes classifier. A Streamlit application is also included so that users can enter a message and receive a classification result in real time. The overall goal is to create a practical and understandable system for spam detection.

## 2. Problem Statement

The problem addressed in this project is the automatic detection of spam messages in text communication. Spam can be harmful, misleading, and disruptive to users. Because spam messages vary widely in wording and structure, the system must learn patterns from historical data rather than rely only on fixed rules.

This project addresses the problem by building a machine learning model that learns from labelled examples and classifies new messages as either spam or ham.

## 3. Objectives

The objectives of this study are:
- To analyze a labelled dataset of messages.
- To preprocess and normalize the text data.
- To convert text into numerical features using TF-IDF.
- To train a Multinomial Naive Bayes classifier.
- To evaluate the performance of the model.
- To develop a simple web-based application for message classification.

## 4. Literature Review

Text classification is one of the most important tasks in natural language processing and machine learning. It involves assigning a text document or message to a category based on its content. Spam detection is a common application of text classification because each message must be classified as either spam or legitimate.

Several machine learning algorithms have been used for spam detection, including Naive Bayes, Support Vector Machines, Logistic Regression, and deep learning models. Among these, Naive Bayes remains very popular because it is efficient, simple, and works well with text data. It is particularly effective when paired with TF-IDF features, which represent the importance of words in a document relative to the entire dataset.

TF-IDF is widely used in text classification because it reduces the influence of common words and increases the value of words that are more informative. This makes it especially useful for distinguishing spam from legitimate messages. The combination of TF-IDF and Naive Bayes is therefore a well-established and effective baseline approach for spam detection.

This project follows this approach by using TF-IDF feature extraction and a Multinomial Naive Bayes classifier. The method is practical, efficient, and suitable for short-text classification tasks such as SMS and email message filtering.

## 5. Methodology

### 5.1 Data Collection
The project uses a dataset of SMS messages labelled as spam or ham. Each record contains the message content and its corresponding class label. This labelled data is used for model training and evaluation.

### 5.2 Data Preprocessing
The data is cleaned before model training. This includes:
- removing missing or empty values,
- converting text to lowercase,
- removing unnecessary whitespace,
- and mapping labels to binary categories.

These steps improve consistency and reduce noise in the dataset.

### 5.3 Feature Extraction with TF-IDF
The cleaned messages are transformed into numerical vectors using TF-IDF. This method measures the importance of words based on their occurrence in a message and their rarity across the entire dataset.

### 5.4 Model Selection
The project uses the Multinomial Naive Bayes classifier. This model is appropriate for text classification because it works efficiently with word-based features and sparse data.

### 5.5 Training and Evaluation
The dataset is divided into training and testing sets using a stratified split. The model is trained on the training data and evaluated on the unseen test data. Performance is measured using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics provide a clear evaluation of the classifier’s ability to detect spam and ham messages.

## 6. System Architecture

The system consists of two major components:

### 6.1 Training Module
The training module handles:
- dataset loading,
- preprocessing,
- feature extraction,
- model training,
- evaluation,
- and saving the trained model and vectorizer.

### 6.2 Application Module
The application module loads the trained model and allows the user to enter a message. The input message is transformed using the saved TF-IDF vectorizer and then passed to the model for classification. The result is displayed with the spam probability.

This design makes the application efficient because the model is trained once and reused repeatedly.

## 7. Implementation

The project is implemented in Python using the following libraries and tools:
- Pandas for dataset handling,
- NumPy for numerical processing,
- scikit-learn for TF-IDF and model training,
- Streamlit for the interactive interface,
- Joblib for saving and loading model files.

The code is organized into a training script and a Streamlit application, making the project easier to maintain and extend.

## 8. Results and Discussion

The evaluation results show that the model performs effectively on the selected dataset. The classifier is able to distinguish spam messages from legitimate ones with reasonable accuracy. The confusion matrix and other metrics provide evidence of the model’s predictive strength.

The TF-IDF and Naive Bayes approach is useful because the model is quick to train, easy to explain, and suitable for short-text classification. It is also practical for academic purposes and small-scale deployment.

However, the project also has limitations. The dataset is based mainly on SMS data, which may not fully represent all types of spam encountered in modern communication systems. In addition, Naive Bayes is a strong baseline model, but more advanced models may produce better results on larger and more complex datasets.

## 9. Conclusion

This project successfully demonstrates the development of a spam detection system using machine learning. By applying TF-IDF and a Multinomial Naive Bayes classifier, the system can classify messages as spam or ham with good accuracy and efficiency. The project also includes a Streamlit interface that makes the system accessible to users without technical knowledge.

Overall, the project is valuable both academically and practically. It demonstrates how machine learning can be used to solve a real-world communication problem and provides a foundation for future improvements in spam detection and text classification systems.

## 10. Recommendations

To improve the system in future work, the following steps may be considered:
- use a larger and more diverse dataset,
- compare Naive Bayes with other classifiers such as Logistic Regression or Support Vector Machines,
- improve text preprocessing and feature extraction,
- and enhance the web application with confidence scores and message history.

These improvements would make the system more robust and more suitable for real-world deployment.

## 11. References

1. UCI Machine Learning Repository, SMS Spam Collection Dataset.
2. scikit-learn Documentation.
3. Python Documentation.
4. Streamlit Documentation.
5. Jurafsky, D., and Martin, J. H. Speech and Language Processing.
6. Aggarwal, C. C., and Zhai, C. Mining Text Data.
