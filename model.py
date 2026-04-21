import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Relu to Leaky Relu is not much of a computation increase, but is better for negative values
        The jump from Leaky Relu to ELU is a much bigger increase, and brings in the domain of GELU, SiLU, etc.
        """
        x = F.leaky_relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    def train_step(self, state, action, reward, next_state, gameOver):
        state = np.array(state, dtype=float)
        state = torch.tensor(state, dtype=torch.float)
        next_state = np.array(next_state, dtype=float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            gameOver = (gameOver, )
        pred =  self.model(state)
        
        # Bellman equation
        # QNew = reward + gamma * max(next_predicted Q value) -> only do this if not gameover
        # pred.clone()
        # preds[argmax(action)] = QNew

        target = pred.clone()
        for i in range(len(gameOver)):
            newQ = reward[i]
            if not gameOver[i]:
                newQ = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
                
            target[i][torch.argmax(action[i]).item()] = newQ

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
