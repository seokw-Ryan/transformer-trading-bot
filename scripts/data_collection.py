
import os
import yfinance as yf
import pandas as pd

# directory = '../data'
# if not os.path.exists(directory):
#     os.makedirs(directory)

def download_data(ticker, period, interval):
    data = yf.download(ticker, period='max', interval=interval)
    data.to_csv(f'./data/{ticker}_data.csv')
    print(f'Data for {ticker} downloaded successfully.')

if __name__ == '__main__':
    download_data('NVDA', 'max','1d')
