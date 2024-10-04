# scripts/evaluate_new_data.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_predictions(predicted_prices_path, actual_prices_path, sequence_length=60):
    # Load predicted prices
    predicted_prices = np.load(predicted_prices_path)

    # Load actual prices
    df = pd.read_csv(actual_prices_path)
    df = df[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    # Get the actual prices corresponding to the predictions
    actual_prices = df['Close'].values[sequence_length:]

    # Ensure the lengths match
    if len(predicted_prices) != len(actual_prices):
        print('Length mismatch between predicted and actual prices.')
        return

    # Calculate evaluation metrics
    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Mean Absolute Error: {mae:.4f}')

    # Save for visualization
    results_df = pd.DataFrame({
        'Date': df.index[sequence_length:],
        'Actual Price': actual_prices,
        'Predicted Price': predicted_prices
    })
    results_df.to_csv('./data/prediction_results.csv', index=False)

if __name__ == '__main__':
    evaluate_predictions('./data/predicted_prices.npy', './data/NVDA_data.csv')
