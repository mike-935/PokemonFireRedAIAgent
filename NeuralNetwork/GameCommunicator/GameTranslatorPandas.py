import pandas as pd
import os 

class GameTranslatorPandas:
    def __init__(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.move_effects = self.create_effect_list()
        self.move_types = self.create_move_types_list()
        
    def translate(self, message, training=False):
        battle_data = list(map(float, message[1:]))
        print("battle_data:", battle_data)
        #battle_data[-1] = battle_data[-1].strip()
        
        #[0.0, 2.0, 14.0, 18.0, 39.0, 19.0, 18.0, 15.0, 15.0, 22.0, 33.0, 0.0, 0.0, 35.0, 95.0, 35.0, 28.0, 23.0, 4.0, 0.0, 100.0, 15.0, 16.0, 149.0, 2.0, 40.0, 100.0, 35.0, 98.0, 
        # 103.0, 0.0, 40.0, 100.0, 27.0, 1.0, 1.0, 
        # 7.0, 4.0, 23.0, 17.0, 9.0, 11.0, 10.0, 16.0, 3.0]
        # Moves are in wrong format?
        # need to know if in doubles and the status of each pokemon
        # maybe include stuff like weather, terrain, etc.
        columns = [
            "p_type1", "p_type2", "p_level", "p_cur_hp", "p_hp", "p_atk", "p_def", "p_spatk", "p_spdef", "p_spd", "p_mvID1", "p_mvID2", "p_mvID3", "p_mvID4", 
            "p_mvEID1", "p_mvEID2", "p_mvEID3", "p_mvEID4", "p_mvType1", "p_mvType2", "p_mvType3", "p_mvType4", "p_mvDmg1", "p_mvDmg2", "p_mvDmg3", "p_mvDmg4",
            "p_mvAcc1", "p_mvAcc2", "p_mvAcc3", "p_mvAcc4", "p_mvPP1", "p_mvPP2", "p_mvPP3", "p_mvPP4",
            "o_type1", "o_type2", "o_level", "o_cur_hp", "o_hp", "o_atk", "o_def", "o_spatk", "o_spdef", "o_spd", "training"
        ]

        if len(battle_data) != len(columns):
            print("incorrect value match between columns")
    
        df = pd.DataFrame([battle_data], columns=columns)
        
        return df

    # Creates the list of Pokémon move effects from the move_effects.txt file.
    # Each index in the list corresponds to the effect of the move at that index.
    def create_effect_list(self):
        effects = []
        effects_path = os.path.join(self.root_dir, "constants/move_effects.txt")
        with open(effects_path, "r") as move_effects_file:
            lines = move_effects_file.readlines()
            for line in lines:
                effect = line.split(" ")[0]
                effects.append(effect)
        return effects

    # Creates the list of Pokémon types from the types.txt file.
    # Each index in the list corresponds to the type of the move at that index.
    def create_move_types_list(self):
        types = []
        types_path = os.path.join(self.root_dir, "constants/types.txt")
        with open(types_path, "r") as types_file:
            lines = types_file.readlines()
            for line in lines:
                move_type = line.split(" ")[0]
                types.append(move_type)
        return types