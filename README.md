
# Semantic Sentiment Classification using NLP & Machine Learning


##  Project Overview

This project focuses on **Semantic Sentiment Classification**, where the goal is to classify text into sentiment categories (Positive / Negative / Neutral) by understanding not just surface-level words, but the **semantic meaning behind the text**.

Unlike traditional keyword-based sentiment models, this system leverages **advanced NLP preprocessing and multiple embedding techniques** to capture contextual meaning and improve classification accuracy.



##  Problem Statement

Human language is highly contextual and ambiguous. Simple rule-based or bag-of-words approaches fail to capture meaning in real-world text such as:

> "I thought the product would be great, but it’s actually not worth the price."

This project addresses the challenge of:

* Understanding semantic meaning in text
* Handling negation and contextual polarity shifts
* Building a robust ML pipeline for sentiment prediction



## Key Features

* End-to-end NLP pipeline (cleaning → vectorization → modeling)
* Comparison of multiple feature engineering techniques:

  * Bag of Words
  * TF-IDF
  * Word Embeddings (Word2Vec / Doc2Vec)
* Multiple ML models evaluated:

  * Logistic Regression
  * Support Vector Machine (SVM)
  * Random Forest
  * XGBoost
* Cross-validation for reliable performance estimation
* Model persistence using `Pickle` for real-time inference



##  System Architecture

1. **Data Preprocessing**

   * Tokenization
   * Stopword removal
   * Stemming/Lemmatization
   * Text normalization

2. **Feature Engineering**

   * BoW / TF-IDF / Word Embeddings

3. **Model Training**

   * Multiple supervised ML classifiers

4. **Evaluation**

   * Accuracy, Precision, Recall, F1-score
   * Cross-validation comparison

5. **Deployment Ready**

   * Serialized model for inference pipeline



##  Results Summary

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 62%      |
| SVM                 | 65%      |
| Random Forest       | 59%      |
| XGBoost             | 69%      |
| Fine tuning XGBoost + w2v              | 69%      |

Best Performing Model: **[Fine tuning XGBoost + w2v ]**


## Tech Stack

* Python 
* Pandas, NumPy
* Scikit-learn
* NLTK
* Gensim (Word Embeddings)
* XGBoost
* Matplotlib / Seaborn (Visualization)



##  Key Learnings

* How semantic representations improve sentiment classification
* Trade-offs between traditional ML vs embedding-based approaches
* Importance of feature engineering over model complexity
* How evaluation metrics reveal model behavior beyond accuracy



##  How to Run This Project

```bash
# Clone repository
git clone https://github.com/jeeva-anand/Semantic-Sentiment-Classification

# Install dependencies
pip install -r requirements.txt

# Run training script
python train.py

# Run inference
python predict.py
```

---

##  Project Structure

```
├── data/
├── notebooks/
├── src/
│   ├── preprocess.py│   
│   ├── predict.py
│   └── train.py
├── app.py
├── requirements.txt
└── README.md
```


## If you like this project

Feel free to star ⭐ the repository and contribute improvements!
