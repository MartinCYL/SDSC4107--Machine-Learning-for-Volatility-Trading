import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
import time
import requests
import json
import sys

def download_data(ticker, start_date, end_date, interval):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data
ticker="AMD"
result=download_data(ticker, "2015-08-01", "2025-03-01", "1d")
result.to_csv(f"{ticker}.csv")
