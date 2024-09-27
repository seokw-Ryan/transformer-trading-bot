
import os
import yfinance as yf
import pandas as pd

# directory = '../data'
# if not os.path.exists(directory):
#     os.makedirs(directory)

def download_data(ticker, period='5y', interval='1d'):
    data = yf.download(ticker, period=period, interval=interval)
    data.to_csv(f'./data/{ticker}_data.csv')
    print(f'Data for {ticker} downloaded successfully.')

if __name__ == '__main__':
    download_data('AAPL')
