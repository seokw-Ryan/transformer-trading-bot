# scripts/deploy_bot.py

import yaml
import time
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame
from trading_logic import trading_decision

def load_config():
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config = load_config()
    alpaca_api = REST(
        key_id=config['alpaca']['api_key'],
        secret_key=config['alpaca']['secret_key'],
        base_url=config['alpaca']['base_url']
    )

    ticker = 'AAPL'

    while True:
        signal = trading_decision(ticker)

        # Get current position
        try:
            position = alpaca_api.get_position(ticker)
            qty = int(position.qty)
        except Exception as e:
            qty = 0

        if signal == 'BUY' and qty == 0:
            # Place buy order
            alpaca_api.submit_order(
                symbol=ticker,
                qty=1,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            print('Placed BUY order for 1 share of', ticker)

        elif signal == 'SELL' and qty > 0:
            # Place sell order
            alpaca_api.submit_order(
                symbol=ticker,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            print(f'Placed SELL order for {qty} shares of', ticker)

        else:
            print('No action taken.')

        # Wait for the next interval
        time.sleep(60 * 60)  # Wait for 1 hour

if __name__ == '__main__':
    main()
