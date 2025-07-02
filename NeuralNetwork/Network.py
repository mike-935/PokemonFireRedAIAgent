import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("battle_data.csv")

effect_id_cols = ["p_mvEID1", "p_mvEID2", "p_mvEID3", "p_mvEID4"]
effect_ids = [df[col].values for col in effect_id_cols]

#Drop player choice col and effect id columns
X = df.drop(columns=["player_choice"] + effect_id_cols).values
y = df["player_choice"].values

#Train test split for X, y, and all effect ids
X_train, X_test, y_train, y_test, eid1_train, eid1_test, eid2_train, eid2_test, eid3_train, eid3_test, eid4_train, eid4_test = train_test_split(X, y, *effect_ids, test_size=0.2, random_state = 37)

#Convert X to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

#Convert y labels to float tensors
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Convert effect ids to tensors
eid_train = [torch.LongTensor(eid) for eid in [eid1_train, eid2_train, eid3_train, eid4_train]]
eid_test = [torch.LongTensor(eid) for eid in [eid1_test, eid2_test, eid3_test, eid4_test]]


class Network(nn.Module):
    def __init__(self, input_size=172, hidden_layer1=10, hidden_layer2=12, output_size = 4):
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