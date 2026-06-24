
import pickle
import pandas as pd
import numpy as np
import os


from src.preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "XGBClassifier-model.pkl")
VECTOR_PATH = os.path.join(BASE_DIR, "..", "models", "word2vec-model.pkl")

MODEL_PATH = os.path.normpath(MODEL_PATH)
VECTOR_PATH = os.path.normpath(VECTOR_PATH)


@st.cache_resource
def load_models():
    with open(MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)

    with open(VECTOR_PATH, "rb") as f:
        vector_model = pickle.load(f)

    return ml_model, vector_model


ml_model, vector_model = load_models()

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
