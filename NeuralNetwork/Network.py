import torch 
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_size=44, hidden_layer1=10, hidden_layer2=12, output_size = 4):
        super().__init__()
        self.connection1 = nn.Linear(input_size, hidden_layer1)
        self.connection2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.out = nn.Linear(hidden_layer2, output_size)

    def forward(self, x):
        x = F.relu(self.connection1(x))
        x = F.relu(self.connection2(x))
        x = self.out(x)
        return x
    

torch.manual_seed(37) 
model = Network()