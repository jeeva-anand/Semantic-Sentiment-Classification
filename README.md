#  Sentiment Analysis using NLP & Machine Learning

##  Project Overview

This project focuses on Sentiment Analysis using various Natural Language Processing (NLP) techniques and Machine Learning models to classify text sentiment as positive or negative.

The main objective of this project was to explore how different text representation techniques impact model performance.


## The project includes:

Text preprocessing
Stemming & Lemmatization
Feature extraction techniques
Multiple machine learning algorithms
Model comparison and evaluation


## Technologies Used

Python
Pandas
NumPy
Scikit-learn
NLTK
Gensim
XGBoost
Matplotlib
Seaborn

## NLP Techniques Used

### Text Preprocessing

The dataset was cleaned using:

Lowercasing
Removing punctuation
Removing stopwords
Tokenization


## Stemming

Words were reduced to their root forms using stemming.

Example:

playing → play
liked → like

This helped reduce vocabulary size.

## Lemmatization

Lemmatization was applied to convert words into meaningful base forms.

Example:

better → good
running → run

Compared to stemming, lemmatization preserves proper word meaning.

## Feature Extraction Techniques

The following text vectorization techniques were explored:

### 1. Bag of Words (BoW)

Converts text into frequency-based vectors.

### 2. TF-IDF

Measures word importance based on frequency and rarity across documents.

### 3. Word Vectors

Word embeddings were used to capture semantic relationships between words.

This approach achieved the best performance of 84% accuracy.

### 4. Document Vectors

Document-level embeddings were used to represent complete sentences/documents numerically.

## Machine Learning Models Used

The following models were trained and evaluated:

| Model                        | Purpose                              |
| ---------------------------- | ------------------------------------ |
| Logistic Regression          | Baseline linear classifier           |
| Support Vector Machine (SVM) | High-dimensional text classification |
| Random Forest                | Ensemble learning                    |
| XGBoost                      | Boosting-based optimized classifier  |


## Best Result

| Technique   | Model                              | Accuracy |
| ----------- | ---------------------------------- | -------- |
| Word Vector | XGBoost / SVM *(edit accordingly)* | **84%**  |


## Project Workflow
1. Data Collection
2. Text Cleaning
3. Stemming & Lemmatization
4. Feature Extraction
5. Model Training
6. Hyperparameter Tuning
7. Model Evaluation
8. Performance Comparison
9. Evaluation Metrics


## The models were evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix


## Key Learnings

Through this project, I learned:

How NLP preprocessing impacts model performance
Differences between BoW, TF-IDF, Word2Vec, and Doc2Vec
How different machine learning algorithms behave on text data
Importance of feature engineering in NLP tasks
Model tuning and evaluation techniques
Future Improvements

## Possible future enhancements include:

Deep Learning models (LSTM, GRU)
Transformer-based models (BERT)
Deployment using Flask or Streamlit
Real-time sentiment prediction web app


## 🔗 Author

**Anand R**
>  This project is part of my journey to becoming a Machine Learning Engineer.
