

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import os
import sys


@dataclass
class DataIngestionConfig:
  raw_data_path = os.path.join('artifacts', 'data.csv')
  train_data_path = os.path.join('artifacts', 'train.csv')
  test_data_path = os.path.join('artifacts', 'test')


class DataIngestion:
  def __init__(self):
    self.data_ingestion_obj = DataIngestionConfig()

  def initiate_data_ingestion(self):
    logging.info("Entered the data ingestion function")

    try:

      train = pd.read_csv('')
      test = pd.read_csv('')
      raw = pd.concat([train, test], ignore_index=True)

      logging.info("Read the dataset as dataframe")

      os.makedirs(os.join.dirname(
          self.data_ingestion_ojb.raw_data_path), exist_ok=True)

      raw.to_csv(self.data_ingestion_ojb.raw_data_path,
                 index=False, header=True)
      train.to_csv(self.data_ingestion_ojb.train_data_path,
                   index=False, header=True)
      test.to_csv(self.data_ingestion_ojb.test_data_path,
                  index=False, header=True)

      return (self.data_ingestion_ojb.train_data_path, self.data_ingestion_ojb.test_data_path)

    except Exception as e:
      raise CustomException(e, sys)


if __name__ == "__main__":
  obj = DataIngestion()
  obj.initiate_data_ingestion()