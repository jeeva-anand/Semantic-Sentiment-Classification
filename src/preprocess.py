
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import pandas as pd
from src.data_loader import data_loader
import re
from nltk import PorterStemmer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_tweet(tweet):
    # Remove mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove non-alphabetic characters except hashtags
    tweet = re.sub(r'[^a-zA-Z#]', ' ', tweet)
    # Remove short words (3 characters or less)
    tweet = ' '.join([w for w in tweet.split() if len(w) > 3])
    return tweet


def merge_data(train_file, test_file):
    train = data_loader(train_file)
    test = data_loader(test_file)
    
    comb = pd.concat([train, test], ignore_index=True)
    
    return comb

def split_data(data):
    tokenize_tweet = data['new_tweet'].str.split(" ")
    return tokenize_tweet


def stemming_tweet(tokenize_tweet):
   
    ps = PorterStemmer()
    tokenize_tweet = tokenize_tweet.apply(lambda x: ([ps.stem(w) for w in x]))         
    tokenize_tweet.apply(
        " ".join(tokenize_tweet[i]) for i in range(len(tokenize_tweet)))
    return tokenize_tweet
    
    

def preprocess_data(data):
  
    
    train = data_loader('../data/train_E6oV3lV.csv')
    test = data_loader('../data/test_tweets_anuFYb8.csv')
    
    comb = merge_data(train,test)
    
    comb['new_tweet'] = comb['tweet'].apply(lambda x: clean_tweet(x))
    
    tokenize_tweet = split_data(comb)
    
    

    

    
    return tokenize_tweet


data = data_loader('../data/test_tweets_anuFYb8.csv')

df = preprocess_data(data)

print(df.hea)