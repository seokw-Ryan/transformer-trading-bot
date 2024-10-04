# scripts/predict_new_data.py

import torch
import numpy as np
from models.transformer_model import TransformerTimeSeries

def predict_new_data(x_new_path, model_path, scaler_path):
    # Load the preprocessed new data
    x_new = np.load(x_new_path)
    x_new_tensor = torch.from_numpy(x_new).float().unsqueeze(2)  # Shape: [samples, seq_len, 1]

    # Adjust shape for the transformer input: [seq_len, batch_size, feature_size]
    x_new_tensor = x_new_tensor.permute(1, 0, 2)

    # Load the model
    model = TransformerTimeSeries()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Make predictions
    with torch.no_grad():
        predictions = model(x_new_tensor)
        predictions = predictions.view(-1).numpy()

    # Load the scaler to inverse transform
    scaler_value = np.load(scaler_path)
    predicted_prices = predictions * scaler_value  # Rescale to original prices

    # Save the predictions
    np.save('./data/predicted_prices.npy', predicted_prices)
    print('Predictions made and saved.')

if __name__ == '__main__':
    predict_new_data('./data/x_data.npy', './models/transformer_model.pth', './data/scaler.npy')
