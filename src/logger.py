
import logging
import os
from datetime import datetime


FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_file_path = os.path.join(os.getcwd(), "logs", FILE_NAME)
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    logging.info(
        "Logging setup complete. Log file created at: %s", log_file_path)
