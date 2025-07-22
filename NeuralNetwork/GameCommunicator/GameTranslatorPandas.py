import pandas as pd
import os 

class GameTranslatorPandas:
    def __init__(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        #self.move_effects = self.create_effect_list()
        #self.move_types = self.create_move_types_list()
        
    def translate(self, message):
        print(f"[TRANSLATE] translate() called., Command is {message[0]}")
        battle_data = list(map(float, message[1:]))
        #print("battle_data:", battle_data)
        #battle_data[-1] = battle_data[-1].strip()

        #[0.0, 2.0, 14.0, 18.0, 39.0, 19.0, 18.0, 15.0, 15.0, 22.0, 33.0, 0.0, 0.0, 35.0, 95.0, 35.0, 28.0, 23.0, 4.0, 0.0, 100.0, 15.0, 16.0, 149.0, 2.0, 40.0, 100.0, 35.0, 98.0,
        # 103.0, 0.0, 40.0, 100.0, 27.0, 1.0, 1.0, 
        # 7.0, 4.0, 23.0, 17.0, 9.0, 11.0, 10.0, 16.0, 3.0]
        # Moves are in wrong format?
        # need to know if in doubles and the status of each pokemon
        # maybe include stuff like weather, terrain, etc.
        columns = [
            "p_type1", "p_type2", "p_level", "p_ability" , "p_status", "p_cur_hp", "p_hp", "p_atk", "p_def", "p_spatk", "p_spdef", "p_spd",
            "p_mvID1", "p_mvEID1", "p_mvType1", "p_mvDmg1", "p_mvAcc1", "p_mvPP1", 
            "p_mvID2", "p_mvEID2", "p_mvType2", "p_mvDmg2", "p_mvAcc2", "p_mvPP2", 
            "p_mvID3", "p_mvEID3", "p_mvType3", "p_mvDmg3", "p_mvAcc3", "p_mvPP3",
            "p_mvID4", "p_mvEID4", "p_mvType4", "p_mvDmg4", "p_mvAcc4", "p_mvPP4",
            "o_type1", "o_type2", "o_level", "o_ability" , "o_status", "o_cur_hp", "o_hp", "o_atk", "o_def", "o_spatk", "o_spdef", "o_spd",
            "p2_type1", "p2_type2", "p2_level", "p2_ability" , "p2_status", "p2_cur_hp", "p2_hp", "p2_atk", "p2_def", "p2_spatk", "p2_spdef", "p2_spd", "p2_switchable",
            "p3_type1", "p3_type2", "p3_level", "p3_ability" , "p3_status", "p3_cur_hp", "p3_hp", "p3_atk", "p3_def", "p3_spatk", "p3_spdef", "p3_spd", "p3_switchable",
            "p4_type1", "p4_type2", "p4_level", "p4_ability" , "p4_status", "p4_cur_hp", "p4_hp", "p4_atk", "p4_def", "p4_spatk", "p4_spdef", "p4_spd", "p4_switchable",
            "p5_type1", "p5_type2", "p5_level", "p5_ability" , "p5_status", "p5_cur_hp", "p5_hp", "p5_atk", "p5_def", "p5_spatk", "p5_spdef", "p5_spd", "p5_switchable",
            "p6_type1", "p6_type2", "p6_level", "p6_ability" , "p6_status", "p6_cur_hp", "p6_hp", "p6_atk", "p6_def", "p6_spatk", "p6_spdef", "p6_spd", "p6_switchable",
            "player_choice"
        ]

        if len(battle_data) != len(columns):
            print("incorrect value match between columns")
    
        df = pd.DataFrame([battle_data], columns=columns)
        
        #Setting values to normalize each stat by 
        normalization_values = {
            "level": 100,
            "hp":  714,
            "atk": 2016,
            "def": 2456, 
            "spatk": 2016,
            "spdef": 2456,
            "spd": 2016
        }
        
        #normalizing current hp for all pokemon
        df["p_cur_hp"] = df["p_cur_hp"] / df["p_hp"]
        df["o_cur_hp"] = df["o_cur_hp"] / df["o_hp"]
        
        for i in range(2,7):
            if df[f"p{i}_hp"].iloc[0] == 0:
                print(f"Warning: p{i}_hp is zero, setting p{i}_cur_hp to 0 to avoid division by zero.")
                df[f"p{i}_cur_hp"] = 0
                continue
            df[f"p{i}_cur_hp"] = df[f"p{i}_cur_hp"] / df[f"p{i}_hp"]
        
        
        #Grabbing different prefixes for each 
        normalization_columns = ["p", "o"] + [f"p{i}" for i in range(2,7)]
        
        #Normalizing values 
        for normal_col in normalization_columns:
            for col, factor in normalization_values.items():
                stat = f"{normal_col}_{col}"
                if stat in df.columns:
                    df[stat] = df[stat] / factor
        
        
        type_columns = [
            "p_type1", "p_type2", "o_type1", "o_type2"
        ]

        # Add party Pokémon types
        for i in range(2, 7):
            type_columns.append(f"p{i}_type1")
            type_columns.append(f"p{i}_type2")

        type_indexes = {}

        for i in range(18):
            type_indexes[i] = i
            
        df = self.add_one_hot(df, type_columns, type_indexes, 18)

        status_columns = [
            "p_status", "o_status"
        ]

        status_indexes = {
            0: 0,  # No status
            1: 1,  # 1 turn of sleep left
            2: 2,  # 2 turns of sleep left
            3: 3,  # 3 turns of sleep left
            4: 4,  # 4 turns of sleep left
            5: 5,  # 5 turns of sleep left (Unlikely, but possible?)
            6: 6,  # 6 turns of sleep left (Unlikely, but possible?)
            7: 7,  # 7 turns of sleep left (Unlikely, but possible?)
            8: 8,  # Poison
            16: 9, # Burn
            32: 10, # Frozen
            64: 11, # Paralyzed
            128: 12, # Toxic
        }

        for i in range(2, 7):
            status_columns.append(f"p{i}_status")

        df = self.add_one_hot(df, status_columns, status_indexes, size=13)
            
        file_path = os.path.join(self.root_dir, "battle_data.csv")
        
        if message[0] == "SAVE_MOVE":
            df.to_csv(file_path, mode="a", header=not os.path.exists(file_path), index=False)
        elif message[0] == "REQUEST_AI_MOVE":
            print(f"Not printing to file, as this is a request for AI move. Data: {df}")
        return df

    def one_hot_encode_data(self, data, indexes, size):
        one_hot = [0] * size
        data_to_encode = int(data)
        if data_to_encode in indexes:
            encoded_index = indexes.get(data_to_encode)
            one_hot[encoded_index] = 1
        return one_hot

    def add_one_hot(self, dataframe, columns, indexes, size):
        for col in columns:
            one_hot_df = dataframe[col].apply(self.one_hot_encode_data, args=(indexes, size)).apply(pd.Series)
            one_hot_df.columns = [f"{col}_{i}" for i in range(size)]
            dataframe.drop(columns=[col], inplace=True)

            dataframe = pd.concat([dataframe, one_hot_df], axis=1)
        return dataframe
    
    def one_hot_encode(self, type_id, num_types=18):
        one_hot = [0] * num_types
        if 0 <= int(type_id) < num_types:
            one_hot[int(type_id)] = 1
        return one_hot
    
    def one_hot_type(self, dataframe, columns):
        for col in columns:
            one_hot_df = dataframe[col].apply(self.one_hot_encode).apply(pd.Series)
            one_hot_df.columns = [f"{col}_{i}" for i in range(18)]
            dataframe.drop(columns=[col], inplace=True)
            
            dataframe = pd.concat([dataframe, one_hot_df], axis=1)
        return dataframe

    # Creates the list of Pokémon move effects from the move_effects.txt file.
    # Each index in the list corresponds to the effect of the move at that index.
    def create_effects_dict(self):
        effects = {}
        effects_path = os.path.join(self.root_dir, "constants/move_effects.txt")
        with open(effects_path, "r") as move_effects_file:
            lines = move_effects_file.readlines()
            for line in lines:
                effect_id = line.split(" ")[1]
                effect_encoding_index = line.split(" ")[2]
                effects.update({effect_id: effect_encoding_index})
        return effects

    def update_old_csv_with_status(self):
        battle_data_csv_file = "battle_data.csv"
        current_directory = os.path.dirname(__file__)
        absolute_battle_data_path = os.path.abspath(
            os.path.join(current_directory, '..', '..', battle_data_csv_file))
        dd = pd.read_csv(absolute_battle_data_path)
        status_columns = [
            "p_status", "o_status"
        ]

        status_indexes = {
            0: 0,  # No status
            1: 1,  # 1 turn of sleep left
            2: 2,  # 2 turns of sleep left
            3: 3,  # 3 turns of sleep left
            4: 4,  # 4 turns of sleep left
            5: 5,  # 5 turns of sleep left (Unlikely, but possible?)
            6: 6,  # 6 turns of sleep left (Unlikely, but possible?)
            7: 7,  # 7 turns of sleep left (Unlikely, but possible?)
            8: 8,  # Poison
            16: 9,  # Burn
            32: 10,  # Frozen
            64: 11,  # Paralyzed
            128: 12,  # Toxic
        }

        for i in range(2, 7):
            status_columns.append(f"p{i}_status")

        dd = self.add_one_hot(dd, status_columns, status_indexes, size=13)
        file_path = os.path.join(self.root_dir, "battle_data4.csv")
        dd.to_csv(file_path, mode="w", header=not os.path.exists(file_path), index=False)
        return None

