# scripts/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path, ticker):
    # Load the data
    df = pd.read_csv(file_path)

    # Select the features you want to use as input
    # Let's use Open, High, Low, Close, Volume
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[['Date'] + features]
    
    # Ensure the date is in datetime format and sort the data
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    # Prepare the target variable (Closing Price)
    target = df['Close'].values

    # Feature Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df[features])

    # Save the scaler parameters
    np.savez('./data/scaler.npz', scale_=scaler.scale_, min_=scaler.min_)

    # Prepare sequences
    sequence_length = 60
    x_data = []
    y_data = []

    # Create input sequences and targets
    for i in range(sequence_length, len(scaled_features)):
        x_data.append(scaled_features[i-sequence_length:i])  # Shape: (sequence_length, number_of_features)
        y_data.append(target[i])  # Original closing price (unscaled)

    x_data = np.array(x_data)  # Shape: (samples, sequence_length, number_of_features)
    y_data = np.array(y_data)  # Shape: (samples,)

    # Save preprocessed data
    np.save(f'./data/{ticker}_data.npy', x_data)
    np.save(f'./data/{ticker}_data.npy', y_data)
    print('Data preprocessed and saved successfully.')

if __name__ == '__main__':
    preprocess_data('./data/NVDA_data.csv', 'NVDA')
