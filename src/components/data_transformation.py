

import sys
import os
import re
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pt
from sklearn.metrics import f1_score
from dataclasses import dataclass
from gensim.models import Word2Vec
from src.exception import CustomException
from src.logger import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from src.utils import save_object
nltk.download('wordnet')
nltk.download('omw-1.4')


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'tfidf.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.stemmer = nltk.PorterStemmer()

    def preprocess_text(self, text):
        try:
            if pd.isna(text):
                return ""
            text = str(text)
            text = re.sub(r"@\w+", "", text)
            text = re.sub(r"[^a-zA-Z#]", " ", text)
            text = " ".join(
                [word for word in text.split() if len(word) > 3]
            )
            words = text.split()
            words = [
                self.stemmer.stem(word)
                for word in words
            ]
            text = " ".join(words)
            return text

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        try:
            tfidf = TfidfVectorizer(
                min_df=1,
                max_df=0.9,
                max_features=1000,
                stop_words="english"
            )

            return tfidf
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
         
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

     
            X_train = train_df["tweet"]
            y_train = train_df["label"]

            X_test = test_df["tweet"]
            y_test = (
                test_df["label"]
                if "label" in test_df.columns
                else None
            )
            X_train = X_train.apply(self.preprocess_text)
            X_test = X_test.apply(self.preprocess_text)
            
            preprocessing_obj = self.get_data_transformer_object()
            
            train_arr = preprocessing_obj.fit_transform(X_train)
            test_arr = preprocessing_obj.transform(X_test)
            
            train_df = pd.concat(
                      [pd.DataFrame(train_arr), pd.DataFrame(y_train)], axis=1)
            test_df = pd.concat([pd.DataFrame(test_arr), pd.DataFrame(y_test)], axis = 1)

            
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessing_obj
            )
            
            
            return (
               train_arr,
               test_arr,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()
    obj.initiate_data_transformation(
        train_path="artifacts/train.csv",
        test_path="artifacts/test.csv"
    )
    