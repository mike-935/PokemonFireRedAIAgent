import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np

class Network(nn.Module):
    def __init__(self, input_size=530, hidden_layer1=10, hidden_layer2=12, output_size = 9):
        super().__init__()
        #self.df = pd.read_csv("battle_data.csv")
        self.move_effect_embeddings = nn.Embedding(355, 16)
        self.ability_embeddings = nn.Embedding(78, 8)
        self.connection1 = nn.Linear(input_size, hidden_layer1)
        self.connection2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.out = nn.Linear(hidden_layer2, output_size)

    def separate_data(self, dataframe):
        # Select the columns that we want to extract
        move_effect_columns = ["p_mvEID1", "p_mvEID2", "p_mvEID3", "p_mvEID4"]
        ability_columns = ["p_ability", "o_ability"]
        # Now we do this for the party Pokémon
        for i in range(2, 7):
            ability_columns.append(f"p{i}_ability")
        # Now that we know the columns we can extract the data
        # We have all the regular data
        # with the move effect ids, and the ability ids separated to be embedded

        numerical_data = dataframe.drop(columns=move_effect_columns + ability_columns
                                                + ["player_choice"], axis=1)

        move_effect_data = dataframe[move_effect_columns].values
        ability_data = dataframe[ability_columns].values
        player_choice_data = dataframe["player_choice"].values
        return numerical_data, move_effect_data, ability_data, player_choice_data

    # To prepare the data to be based into our network,
    # we need to separate the numerical data from the categorical data that we want to embed.
    def format_datasets(self, dataframe):
        (numerical_data, move_effect_data, ability_data,
         player_choice_data) = self.separate_data(dataframe)

        # Once we've extracted we can convert them to tensors to be used in the model
        #numerical_data.to_csv("numerical_data.csv", index=False)
        numerical_tensor = torch.tensor(numerical_data.values, dtype=torch.float32)
        #np.savetxt('readable_tensor.txt', numerical_tensor.numpy(), delimiter=',', fmt='%.4f')
        move_effect_tensor = torch.tensor(move_effect_data, dtype=torch.long)
        ability_tensor = torch.tensor(ability_data, dtype=torch.long)
        player_choice_tensor = torch.tensor(player_choice_data, dtype=torch.long)
        return numerical_tensor, move_effect_tensor, ability_tensor, player_choice_tensor

    def forward(self, x_numerical_tensor, move_effect_tensor,
                ability_tensor):
        x_move_effect_embeddings = self.move_effect_embeddings(move_effect_tensor)
        x_ability_embeddings = self.ability_embeddings(ability_tensor)

        # Concatenate all the tensors, with the embeddings flattened
        x = torch.cat((x_numerical_tensor,
                       x_move_effect_embeddings.view(x_move_effect_embeddings.size(0), -1),
                       x_ability_embeddings.view(x_ability_embeddings.size(0), -1)), dim=1)

        # Pass through the network
        x = F.relu(self.connection1(x))
        x = F.relu(self.connection2(x))
        x = self.out(x)
        return x

    def train_test_network(self):
        battle_data_csv_file = "battle_data.csv"
        current_directory = os.path.dirname(__file__)
        absolute_battle_data_path = os.path.abspath(os.path.join(current_directory, '..', battle_data_csv_file))
        if not os.path.exists(absolute_battle_data_path):
            print(f"Training skipped: '{battle_data_csv_file}' not found.")
            return
        
        print("Training the network...")
        self.train()
        df = pd.read_csv(absolute_battle_data_path)

        (training_numerical, training_move_effects,
         training_abilities, training_player_choice) = self.format_datasets(df)

        (x_training_numerical, x_testing_numerical,
         x_training_move_effects, x_testing_move_effects,
         x_training_abilities, x_testing_abilities,
         y_training_player_choice, y_testing_player_choice) = (
            train_test_split(training_numerical, training_move_effects,
                             training_abilities,
                             training_player_choice, test_size=0.2, random_state=42))
        print("Used Sklearn's train_test_split the data into "
              "(x_training_numerical, x_testing_numerical, x_training_move_effects, "
              "x_testing_move_effects, x_training_abilities, x_testing_abilities, "
              "y_training_player_choice, "
              "y_testing_player_choice.")
        # take out training data plus our move effects, abilities,
        # and lastly the player choice
        # And wrap it into a dataset which is just a table for each row
        training_dataset = torch.utils.data.TensorDataset(
            x_training_numerical, x_training_move_effects,
            x_training_abilities, y_training_player_choice)
        print("Created a training dataset with the training data and player choice.")
        # Now with that dataset we can put them in randomized groups of 64
        training_loader = DataLoader(training_dataset, batch_size=64, shuffle=True)
        print("Created a training loader with the training dataset with batches of suze 64")

        # Now we do the same for the testing data
        testing_dataset = torch.utils.data.TensorDataset(x_testing_numerical,
                                                         x_testing_move_effects,
                                                         x_testing_abilities,
                                                         y_testing_player_choice)
        print("Created a testing dataset with the testing data and player choice.")
        testing_loader = DataLoader(testing_dataset, batch_size=64, shuffle=False)

        print("Created a testing loader with the testing dataset with batches of size 64")


        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        epochs = 100
        losses = []
        print("testing for NaaN")
        print("Any NaNs in numerical data:", torch.isnan(x_training_numerical).any().item())
        print("Any NaNs in move effects:", torch.isnan(x_training_move_effects).any().item())
        print("Any NaNs in abilities:", torch.isnan(x_training_abilities).any().item())
        print("Any NaNs in player choices:", torch.isnan(y_training_player_choice).any().item())

        print(f"Starting training for {epochs} epochs...")
        # For each epoch we will go through the training data and train
        for i in range(epochs):
            # for each group of 64 rows of data we will train the network
            # and compare the output to the player choice we had recorded
            for batch in training_loader:
                (x_numerical_tensor, move_effect_tensor,
                 ability_tensor, player_choice_tensor) = batch

                # clear previous gradients
                optimizer.zero_grad()

                # get what the network thinks the correct output should be for the data
                output = self(x_numerical_tensor, move_effect_tensor,
                                      ability_tensor)

                # calculate the loss between what the network thinks is correct and
                # what we the player actually chose
                loss = criterion(output, player_choice_tensor)

                losses.append(loss.detach().numpy())
                # Now we can backpropagate the loss and update the network
                # so we can backpropagate and adjust the neurons' weights and biases
                # to minimize the loss (computing gradients)
                loss.backward()

                # Apply gradients and update the weights and bias
                optimizer.step()

            if i % 10 == 0:
                print(f'Epoch [{i + 1}/{epochs}], Loss: {losses[i]:.4f}')

        print("Training complete.")
        self.eval()
        print("Starting testing...")
        correct_choices = 0
        total_choices = 0
        with torch.no_grad():
            print("With torch.no_grad, starting testing...")
            test_index = 0
            # Now we will go through the testing data and see how well the network performs
            # We will compare with our testing set of data
            # Like with training we will go through the data in groups of 64
            # And take that group's battle data, plug that into the network
            # nd get what the network thinks the player should choose
            # Then compare that to what the testing data had that we chose
            for batch in testing_loader:
                test_index += 1
                (x_numerical_tensors, move_effect_tensors,
                 ability_tensors, player_choices) = batch
                # now lets get the scores for the 0-8 choices
                # the network has decided for the data
                outputs = self(x_numerical_tensors, move_effect_tensors,
                                      ability_tensors,)
                # Now lets turn that list of batch sized 0-8 scores
                # into a batch sized list of the network's predicted choices
                predicted_choices = outputs.argmax(dim=1)
                total_choices += len(predicted_choices)
                correct_choices += (predicted_choices == player_choices).sum().item()
                for actual, predicted in zip(player_choices, predicted_choices):
                    print(f"Actual Choice: {actual.item()}, Network Predicted Choice: {predicted.item()}")

        accuracy = correct_choices / total_choices
        print(f"The network made {correct_choices} correct choices "
              f"out of {total_choices} total choices. With an accuracy of {accuracy:.2%}")


    def generate_ai_decision(self, formatted_battle_data):
        (numerical_tensor, move_effect_tensor, ability_tensor,
         player_choice_placeholder_tensor) = self.format_datasets(formatted_battle_data)
        ai_decision = None
        with torch.no_grad():
            self.eval()
            ai_decision = self(numerical_tensor, move_effect_tensor, ability_tensor)
            print(f"AI Decision Scores: {ai_decision}")
        # Get the index of the maximum score
        ai_decision_max = ai_decision.argmax()
        print(f"AI Decision Index: {ai_decision_max} and the .item() is {ai_decision_max.item()}")
        return ai_decision_max.item()
