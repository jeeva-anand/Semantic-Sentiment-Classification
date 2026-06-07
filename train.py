import pickle
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec

from preprocess import clean_tweet
from preprocess import word_vector
from model import train_model


# train = pd.read_csv("./data/train_E6oV3lV.csv")
# test = pd.read_csv("./data/test_tweets_anuFYb8.csv")

# df = pd.concat([train, test], ignore_index=True)

# df["clean_tweet"] = df["tweet"].apply(clean_tweet)

# vectorizer = TfidfVectorizer(
#     min_df=1,
#     max_df=0.9,
#     max_features=1000,
#     stop_words="english"
# )


# X = df["clean_tweet"]

# y = df["label']

# X = vectorizer.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.3,
#     random_state=42
# )

# model = LogisticRegression()

# model.fit(X_train, y_train)

# pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, pred))

# pickle.dump(model, open("models/sentiment_model.pkl", "wb"))

# pickle.dump(
#     vectorizer,
#     open("models/word2vec_vectorizer.pkl", "wb")
# )

# print("Saved Successfully")

train = pd.read_csv("./data/train_E6oV3lV.csv")
test = pd.read_csv("./data/test_tweets_anuFYb8.csv")

df = pd.concat([train, test], ignore_index=True)

df["clean_tweet"] = df["tweet"].apply(clean_tweet)

tokenize_tweet = df['clean_tweet'].apply(lambda x: x.split())

vectorizer = Word2Vec(tokenize_tweet, vector_size=200, window=5,min_count=2, sg=1, hs=0, negative=10, workers=2, seed=34)

word_vector_arr = np.zeros((len(tokenize_tweet), 200))

for i in range(len(tokenize_tweet)):
  word_vector_arr[i] = word_vector(tokenize_tweet[i], 200, vectorizer)

word_vec_df = pd.DataFrame(word_vector_arr)

train_w2v =word_vec_df.iloc[:31962, :]

xtrain, xtest, ytrain, ytest = train_test_split(
    train_w2v, train['label'], test_size=0.3, random_state=42)

model = train_model(xtrain, ytrain, xtest, ytest)

pickle.dump(model,open("models/sentiment_model.pkl", "wb"))

pickle.dump(
    vectorizer,
    open("models/word2vec_vectorizer.pkl", "wb")
)

print("Saved Successfully")
