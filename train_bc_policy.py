import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


# 1. Define the PyTorch Dataset
class ExpertDataset(Dataset):
    def __init__(self, data_path):
        print(f"📂 Loading Expert Data from {data_path}...")
        data = np.load(data_path, allow_pickle=True)

        self.states = torch.FloatTensor(data['states'])
        self.actions = torch.FloatTensor(data['actions'])
        self.language = data['language']

        print(f"✅ Loaded {len(self.states)} trajectory steps.")
        print(f"🗣️ Sample Language Command: '{self.language[0]}'")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


# 2. Define the Neural Network Architecture
class BehaviorCloningPolicy(nn.Module):
    def __init__(self, input_dim=6, output_dim=3):
        super().__init__()
        # A lightweight Multilayer Perceptron (MLP) for continuous motor control
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def main():
    # Setup Data
    dataset = ExpertDataset("dataset/vla_expert_data.npz")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize Model, Loss (MSE), and Optimizer
    print("🧠 Initializing Behavior Cloning Neural Network...")
    model = BehaviorCloningPolicy()
    criterion = nn.MSELoss()  # Mean Squared Error is standard for Imitation Learning
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 3. Training Loop
    epochs = 50
    print(f"🚀 Starting Training for {epochs} Epochs...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for states, actions in dataloader:
            # Forward pass
            predictions = model(states)
            loss = criterion(predictions, actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss / len(dataloader):.6f}")

    # 4. Save the trained weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bc_policy.pth")
    print("🎉 Training Complete! Policy weights saved to: models/bc_policy.pth")


if __name__ == "__main__":
    main()