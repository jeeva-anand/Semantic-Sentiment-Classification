from preprocess import clean_tweet
import pickle
import xgboost as xgb

from preprocess import clean_tweet
from preprocess import tokenize_tweets
from preprocess import vectorize_tweet



model = pickle.load(
    open("./models/sentiment_model.pkl", "rb")
)




def predict_sentiment(text):

    text = clean_tweet(text)
    tokenize_tweet = tokenize_tweets(text)     
    word_vec_df = vectorize_tweet(tokenize_tweet)

    dmatrix = xgb.DMatrix(word_vec_df)

    probability = float(model.predict(dmatrix)[0])

    prediction = 1 if probability >= 0.5 else 0


    return prediction, probability
