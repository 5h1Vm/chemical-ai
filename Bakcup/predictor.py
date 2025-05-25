import torch
import torch.nn as nn

# Define your model class (if not already done)
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Define layers here

    def forward(self, x):
        # Define the forward pass
        return x

# Instantiate the model
model = YourModel()

# Load the model's state_dict with weights_only set to False
try:
    model.load_state_dict(torch.load(r"D:/Hope/Detectio/Bakcup/model/color_predictor_model.pkl", map_location=torch.device('cpu'), weights_only=False))
    model.eval()  # Put the model in evaluation mode (if necessary)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
