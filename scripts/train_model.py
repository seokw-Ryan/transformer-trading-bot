# scripts/train_model.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.transformer_model import TransformerTimeSeries

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model():
    x_data = np.load('./data/x_data.npy')
    y_data = np.load('./data/y_data.npy')

    x_tensor = torch.from_numpy(x_data).float().unsqueeze(2)  # Shape: [batch, seq_len, feature_size]
    y_tensor = torch.from_numpy(y_data).float()

    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TransformerTimeSeries()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            x_batch = x_batch.permute(1, 0, 2)  # Transformer expects input shape: [seq_len, batch_size, feature_size]
            output = model(x_batch)
            loss = criterion(output.view(-1), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}')

    # Save the trained model
    torch.save(model.state_dict(), './models/transformer_model.pth')
    print('Model trained and saved successfully.')

if __name__ == '__main__':
    train_model()
