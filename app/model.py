import torch
import torch.nn as nn
import torchvision.transforms as transforms

class SubstanceDetectionModel(nn.Module):
    def __init__(self):
        super(SubstanceDetectionModel, self).__init__()
        # Example: Simple CNN for image classification
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 64 * 64, 512)  # Example flattened size
        self.fc2 = nn.Linear(512, 10)  # Example 10 possible substances
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(path='models/model.pth'):
    model = SubstanceDetectionModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
