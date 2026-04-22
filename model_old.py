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
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    # 1. FIXED: Added target_model to arguments
    def __init__(self, model, target_model, lr, gamma): 
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model 
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, gameOver):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # 3. FIXED: Unsqueeze must happen BEFORE the loop
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            gameOver = (gameOver, )

        pred = self.model(state)
        target = pred.clone()
        
        # 2. FIXED: Removed duplicated old logic, kept only the True DQN update
        for i in range(len(gameOver)):
            newQ = reward[i]
            if not gameOver[i]:
                # TRUE DQN UPDATE: Use the frozen target_model!
                newQ = reward[i] + self.gamma * torch.max(self.target_model(next_state[i]))
            
            target[i][torch.argmax(action[i]).item()] = newQ

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


    def train_step_sarsa(self, state, action, reward, next_state, next_action, gameOver):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        next_action = torch.tensor(next_action, dtype=torch.long) 
        reward = torch.tensor(reward, dtype=torch.float)

        # 4. FIXED: Added the actual unsqueeze logic back in
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            next_action = torch.unsqueeze(next_action, 0)
            reward = torch.unsqueeze(reward, 0)
            gameOver = (gameOver, )

        pred = self.model(state)
        target = pred.clone()
        
        for i in range(len(gameOver)):
            newQ = reward[i]
            if not gameOver[i]:
                # SARSA UPDATE: Use the Q-value of the specific next_action
                next_action_idx = torch.argmax(next_action[i]).item()
                newQ = reward[i] + self.gamma * self.model(next_state[i])[next_action_idx]

            target[i][torch.argmax(action[i]).item()] = newQ

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()