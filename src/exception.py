import sys

from src.logger import logging


def handle_exception(error, error_description: sys):
    _, _, exc_tb = sys.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    lineno = exc_tb.tb_lineno
    error_message = f"Error occurred in file: {filename}, line: {lineno}, error:{error}, description: {error_description}"

    return error_message


class CustomException(Exception):
    
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = handle_exception(error_message, error_details)

    def __str__(self):        
        return self.error_message
