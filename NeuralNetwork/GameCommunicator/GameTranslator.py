import torch
import os

class GameTranslator:
    def __init__(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.move_effects = self.create_effect_list()
        self.move_types = self.create_move_types_list()

    def translate(self, message, training=False):
        battle_data = list(map(float, message[1:]))  # Convert all elements to integers
        formatted_data = []
        # battle_data (command, type1, type2, )
        pkm1Type1 = [0] * 18
        print("Index of Type 1:", int(battle_data[1]))
        pkm1Type1[int(battle_data[0])] = 1
        formatted_data.extend(pkm1Type1)

        pkm1Type2 = [0] * 18
        print("Index of Type 2:", int(battle_data[2]))
        pkm1Type2[int(battle_data[1])] = 1
        formatted_data.extend(pkm1Type2)

        formatted_data.append(battle_data[2] / 100)

        formatted_data.append(battle_data[3] / battle_data[4])

        # Stats
        formatted_data.append(battle_data[4] / 714)
        formatted_data.append(battle_data[5] / 2016)
        formatted_data.append(battle_data[6] / 2456)
        formatted_data.append(battle_data[7] / 2016)
        formatted_data.append(battle_data[8] / 2456)
        formatted_data.append(battle_data[9] / 2016)

        '''
        # battle_data[11] = move id
        # battle_data[12] = move effect id

        moveTypeList1 = [0] * 18
        moveTypeList1[int(battle_data[13])] = 1
        
        battle_data[14] /= 215
        battle_data[15] /= 100
        battle_data[16] = 1 if battle_data[16] > 0 else 0
        '''
        self.format_moves(0, formatted_data, battle_data)

        '''
        # battle_data[17] = move id2
        # battle_data[18] = move effect id2

        moveTypeList2 = [0] * 18
        moveTypeList2[int(battle_data[19])] = 1
        
        battle_data[20] /= 215
        battle_data[21] /= 100
        battle_data[22] = 1 if battle_data[22] > 0 else 0
        '''
        self.format_moves(1, formatted_data, battle_data)

        '''
        # battle_data[23] = move id2
        # battle_data[24] = move effect id2

        moveTypeList3 = [0] * 18
        moveTypeList3[int(battle_data[25])] = 1
        
        battle_data[26] /= 215
        battle_data[27] /= 100
        battle_data[28] = 1 if battle_data[28] > 0 else 0
        '''
        self.format_moves(2, formatted_data, battle_data)

        '''
        # battle_data[29] = move id2
        # battle_data[30] = move effect id2

        moveTypeList4 = [0] * 18
        moveTypeList4[int(battle_data[31])] = 1
        
        battle_data[32] /= 215
        battle_data[33] /= 100
        battle_data[34] = 1 if battle_data[34] > 0 else 0
        '''
        self.format_moves(3, formatted_data, battle_data)

        opponentTypeList = [0] * 18
        opponentTypeList[int(battle_data[34])] = 1
        formatted_data.extend(opponentTypeList)

        opponentTypeList2 = [0] * 18
        opponentTypeList2[int(battle_data[35])] = 1
        formatted_data.extend(opponentTypeList2)

        formatted_data.append(battle_data[36] / 100)
        formatted_data.append(battle_data[37] / battle_data[38])
        formatted_data.append(battle_data[38] / 714)
        formatted_data.append(battle_data[39] / 2016)
        formatted_data.append(battle_data[40] / 2456)
        formatted_data.append(battle_data[41] / 2016)
        formatted_data.append(battle_data[42] / 2456)
        formatted_data.append(battle_data[43] / 2016)

        if training:
            formatted_data.append(battle_data[44])

        return formatted_data

    def format_moves(self, index, formatted_data, battle_data):
        offset = 6 * index
        # Skip these for now
        # battle_data[10 + offset] = move id
        # battle_data[11 + offset] = move effect id

        move_type_list = [0] * 18
        move_type_list[int(battle_data[12 + offset])] = 1
        formatted_data.extend(move_type_list)

        formatted_data.append(battle_data[13 + offset] / 215)
        formatted_data.append(battle_data[14 + offset] / 100)
        formatted_data.append(1 if battle_data[15 + offset] > 0 else 0)

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
