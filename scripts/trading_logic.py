# scripts/trading_logic.py

import torch
import numpy as np
import pandas as pd
from models.transformer_model import TransformerTimeSeries

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    model = TransformerTimeSeries()
    model.load_state_dict(torch.load('../models/transformer_model.pth'))
    model.eval()
    return model

def get_latest_data(ticker, sequence_length=60):
    df = yf.download(ticker, period='75d', interval='1d')
    df = df[['Close']]
    df = df[-sequence_length:]
    return df

def make_prediction(model, input_sequence):
    input_tensor = torch.from_numpy(input_sequence).float().unsqueeze(1).unsqueeze(2)
    input_tensor = input_tensor.permute(1, 0, 2)
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = prediction.view(-1).numpy()
    return prediction[-1]

def generate_signal(predicted_price, current_price):
    if predicted_price > current_price:
        return 'BUY'
    elif predicted_price < current_price:
        return 'SELL'
    else:
        return 'HOLD'

def trading_decision(ticker):
    # Load scaler
    scaler = np.load('../data/scaler.npy')
    
    # Get latest data
    df = get_latest_data(ticker)
    current_price = df['Close'].iloc[-1]
    input_sequence = df['Close'].values

    # Normalize input sequence
    input_sequence_scaled = input_sequence / scaler[0]

    # Load model
    model = load_model()

    # Make prediction
    predicted_price_scaled = make_prediction(model, input_sequence_scaled)
    predicted_price = predicted_price_scaled * scaler[0]

    # Generate trading signal
    signal = generate_signal(predicted_price, current_price)
    print(f'Predicted Price: {predicted_price:.2f}, Current Price: {current_price:.2f}, Signal: {signal}')
    return signal

if __name__ == '__main__':
    import yfinance as yf
    trading_decision('AAPL')
