import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Linear_Net(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size//2)
        self.linear2 = nn.Linear(hidden_size//2, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        target_folder = os.path.join(os.getcwd(), model_folder_path)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder, exist_ok=True)

        file_name = os.path.join(target_folder, file_name)
        torch.save(self.state_dict(), file_name)
