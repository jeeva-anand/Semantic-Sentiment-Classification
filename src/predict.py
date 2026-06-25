
import pickle
import pandas as pd
import numpy as np
import os
import streamlit as st


from src.preprocess import clean_text

# ml_model = pickle.load(open("models/XGBClassifier-model.pkl", "rb"))

ml_model = joblib.load(open("https://drive.google.com/file/d/1QwU64XxSHMKdRTANqriuc2ejU0rjAe7j/view?usp=sharing", "rb"))
vector_model = pickle.load(open("https://drive.google.com/file/d/1SqIWLY-p02OAe1TvOrdrNjaXmTJw1JsG/view?usp=drive_link", "rb"))



def word_vector(token, size):
  vec = np.zeros(size).reshape((1, size))
  count = 0
  for x in token:
    try:
      vec += vector_model.wv[x].reshape((1, size))
      count += 1
    except KeyError:
      pass

  if count != 0:
    vec /= count

  return vec

def predict_sentiment(text):

    text = clean_text(text)
    
    word_vector_arr = np.zeros((len(text), 200))
    
    word_vector_arr = word_vector(text, 200)

    word_vec_df = pd.DataFrame(word_vector_arr)
   
    probability = ml_model.predict_proba(word_vec_df)
    
    prediction = (probability[:, 1] >= 0.3).astype(int)
    
    print(prediction, probability)
    

    return prediction, probability
