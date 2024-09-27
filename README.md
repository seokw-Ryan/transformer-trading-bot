# Trading Bot Using Transformer Models

## Overview
A Python-based trading bot that utilizes transformer models for stock price prediction and automated trading decisions. The bot leverages the Alpaca API for paper trading and performs buy/sell operations based on model predictions.

## Project Structure
```
trading_bot_transformer/
├── data/                   # Directory for raw and preprocessed data
├── models/                 # Transformer model architecture
├── scripts/                # Scripts for data processing, training, and bot deployment
├── utils/                  # Utility functions
├── config/                 # Configuration files for API keys
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/trading_bot_transformer.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd trading_bot_transformer
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up API keys:** Add your Alpaca API keys in `config/config.yaml`.

## How to Use
1. **Download data:**
   ```bash
   python scripts/data_collection.py
   ```
2. **Preprocess data:**
   ```bash
   python scripts/data_preprocessing.py
   ```
3. **Train the model:**
   ```bash
   python scripts/train_model.py
   ```
4. **Deploy the trading bot:**
   ```bash
   python scripts/deploy_bot.py
   ```

## Trading Logic
The bot generates BUY, SELL, or HOLD signals based on predictions made by the transformer model and places orders accordingly using the Alpaca API.

## Disclaimer
This project is for educational purposes only. Use it at your own risk. Always test thoroughly before using real money for trading.