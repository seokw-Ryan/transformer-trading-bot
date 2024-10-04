# Load the model and inspect its architecture
import torch
from models.transformer_model import TransformerTimeSeries

# Initialize the model
model = TransformerTimeSeries()

# Load the trained weights
model.load_state_dict(torch.load('../models/transformer_model.pth'))

# Print model architecture
print(model)



# Total number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")


#Check the weights of the first layer
print(model.encoder_layer.self_attn.in_proj_weight)