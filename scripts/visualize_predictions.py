# scripts/visualize_predictions.py

import pandas as pd
import matplotlib.pyplot as plt

def visualize_predictions(results_csv_path):
    # Load the results
    results_df = pd.read_csv(results_csv_path)
    results_df['Date'] = pd.to_datetime(results_df['Date'])
    results_df.set_index('Date', inplace=True)

    # Plot the actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Actual Price'], label='Actual Price', color='blue')
    plt.plot(results_df['Predicted Price'], label='Predicted Price', color='red')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    visualize_predictions('./data/prediction_results.csv')
