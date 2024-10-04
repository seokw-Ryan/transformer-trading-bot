# scripts/test_model_on_new_data.py

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

from models.transformer_model import TransformerTimeSeries

# Function to collect new data
def collect_new_data(ticker, period='1mo', interval='1d'):
    import yfinance as yf
    print(f"Collecting new data for {ticker}...")
    data = yf.download(ticker, period=period, interval=interval)
    if not data.empty:
        data.to_csv(f'./data/{ticker}_data.csv')
        print(f"New data for {ticker} saved successfully.")
    else:
        print("Failed to retrieve data.")

# Function to preprocess new data
def preprocess_new_data(file_path, scaler_path, sequence_length=60):
    print("Preprocessing new data...")
    # Load new data
    df = pd.read_csv(file_path)
    df = df[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    # Load the scaler used during training
    scaler_value = np.load(scaler_path)

    # Normalize the data using the scaler
    scaled_data = df['Close'].values.reshape(-1, 1) / scaler_value

    # Prepare sequences
    x_new = []
    for i in range(sequence_length, len(scaled_data)):
        x_new.append(scaled_data[i-sequence_length:i, 0])

    x_new = np.array(x_new)
    dates = df.index[sequence_length:]
    return x_new, df['Close'].values[sequence_length:], dates

# Function to make predictions
def make_predictions(x_new, model_path, scaler_path):
    print("Making predictions...")
    # Load the model
    model = TransformerTimeSeries()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Convert new data to tensor
    x_new_tensor = torch.from_numpy(x_new).float().unsqueeze(2)  # Shape: [samples, seq_len, 1]
    x_new_tensor = x_new_tensor.permute(1, 0, 2)  # Shape: [seq_len, batch_size, feature_size]

    # Make predictions
    with torch.no_grad():
        predictions = model(x_new_tensor)
        predictions = predictions.view(-1).numpy()

    # Load the scaler to inverse transform
    scaler_value = np.load(scaler_path)
    predicted_prices = predictions * scaler_value  # Rescale to original prices
    return predicted_prices

# Function to evaluate and visualize the results
def evaluate_and_visualize(actual_prices, predicted_prices, dates, ticker):
    print("Evaluating and visualizing results...")
    # Create a DataFrame for easy plotting
    results_df = pd.DataFrame({
        'Date': dates,
        'Actual Price': actual_prices,
        'Predicted Price': predicted_prices
    })
    results_df.set_index('Date', inplace=True)

    # Calculate evaluation metrics
    mse = np.mean((actual_prices - predicted_prices) ** 2)
    mae = np.mean(np.abs(actual_prices - predicted_prices))
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Actual Price'], label='Actual Price', color='blue')
    plt.plot(results_df['Predicted Price'], label='Predicted Price', color='red')
    plt.title(f'Actual vs Predicted Prices for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Set the x-axis date format
    ax = plt.gca()  # Get current axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjust date ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format: Year-Month-Day

    plt.xticks(rotation=45)  # Rotate date labels for better readability

    # Save the plot to a folder
    plots_folder = './plots'
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    plot_file_path = f'{plots_folder}/{ticker}_price_prediction.png'
    plt.savefig(plot_file_path)
    print(f"Plot saved to {plot_file_path}")

def main():
   
    #Ask the user for the stock ticker
    ticker = input("Enter the stock ticker to evaluate (e.g., AAPL, MSFT): ")

    #Parameters
    period = '3mo' #Period of data to retrieve 
    interval = '1d' #for daily data
    sequence_length = 60 #Same as used during training

    # Paths
    new_data_path = f'./data/{ticker}_data.csv'
    scaler_path = './data/scaler.npy'
    model_path = './models/transformer_model.pth'

    # Step 1: Collect new data
    collect_new_data(ticker, period=period, interval=interval)

    # Step 2: Preprocess new data
    x_new, actual_prices, dates = preprocess_new_data(new_data_path, scaler_path, sequence_length)

    # Ensure we have data to work with
    if x_new.size == 0:
        print("Not enough data to make predictions.")
        return

    # Step 3: Make predictions
    predicted_prices = make_predictions(x_new, model_path, scaler_path)

    # Step 4: Evaluate and visualize
    evaluate_and_visualize(actual_prices, predicted_prices, dates, ticker)

if __name__ == '__main__':
    main()
