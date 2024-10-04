# scripts/trading_logic.py

import torch
import numpy as np
import yfinance as yf
from models.transformer_model import TransformerTimeSeries

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(feature_size):
    model = TransformerTimeSeries(feature_size=feature_size).to(device)
    model.load_state_dict(torch.load('../models/transformer_model.pth'))
    model.eval()
    return model

def get_latest_data(ticker, sequence_length=60):
    df = yf.download(ticker, period='75d', interval='1d')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df[-sequence_length:]
    return df

def make_prediction(model, input_sequence):
    input_tensor = torch.from_numpy(input_sequence).float().unsqueeze(1).to(device)  # Shape: [sequence_length, 1, feature_size]
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = prediction.view(-1).cpu().numpy()
    return prediction[-1]

def generate_signal(predicted_price, current_price):
    if predicted_price > current_price:
        return 'BUY'
    elif predicted_price < current_price:
        return 'SELL'
    else:
        return 'HOLD'

def trading_decision(ticker):
    # Load scaler parameters
    scaler_params = np.load('../data/scaler.npz')
    scale_ = scaler_params['scale_']
    min_ = scaler_params['min_']

    # Get latest data
    df = get_latest_data(ticker)
    current_price = df['Close'].iloc[-1]
    input_sequence = df.values  # Shape: (sequence_length, number_of_features)

    # Normalize input sequence
    input_sequence_scaled = (input_sequence - min_) / scale_

    # Load model
    feature_size = input_sequence.shape[1]
    model = load_model(feature_size)

    # Make prediction
    predicted_price = make_prediction(model, input_sequence_scaled)

    # No need to inverse transform since we're predicting the actual price
    signal = generate_signal(predicted_price, current_price)
    print(f'Predicted Price: {predicted_price:.2f}, Current Price: {current_price:.2f}, Signal: {signal}')
    return signal

if __name__ == '__main__':
    trading_decision('AAPL')
