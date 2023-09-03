import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network model
class MoveToGoalModel(nn.Module):
    def __init__(self):
        super(MoveToGoalModel, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 2)  # Output: x and z movement

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model
model = MoveToGoalModel()

# Create synthetic data for training
# Assume each sample consists of [agent_x, agent_z, goal_x, goal_z]
data = torch.rand(100, 4)
targets = torch.rand(100, 2)  # x and z movement

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()



# Save the trained model in ONNX format
dummy_input = torch.randn(1, 4)  # Create a dummy input for tracing
onnx_path = "movetogoal_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)
