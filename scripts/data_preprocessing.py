# scripts/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    # Feature Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Save the scaler for future use
    np.save('./data/scaler.npy', scaler.scale_)

    # Prepare sequences
    sequence_length = 60
    x_data = []
    y_data = []

    for i in range(sequence_length, len(scaled_data)):
        x_data.append(scaled_data[i-sequence_length:i, 0])
        y_data.append(scaled_data[i, 0])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Save preprocessed data
    np.save('./data/x_data.npy', x_data)
    np.save('./data/y_data.npy', y_data)
    print('Data preprocessed and saved successfully.')

if __name__ == '__main__':
    preprocess_data('./data/NVDA_data.csv')
