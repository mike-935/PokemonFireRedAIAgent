import torch

class GameTranslator:
    def translate(self, message):
        pass
        # message (command, type1, type2, )
        typeList1 = [0] * 18
        typeList1[int(message[1])] = 1

        typeList2 = [0] * 18
        typeList2[int(message[1])] = 1

        message[3] /= 100
        message[4] /= message[5]
        message[5] /= 714
        message[6] /= 2016
        message[7] /= 2456
        message[8] /= 2016
        message[9] /= 2456
        message[10] /= 2016

        # message[11] = move id
        # message[12] = move effect id

        moveTypeList1 = [0] * 18
        moveTypeList1[int(message[13])] = 1
        
        message[14] /= 215
        message[15] /= 100
        message[16] = 1 if message[16] > 0 else 0
  
        # message[17] = move id2
        # message[18] = move effect id2

        moveTypeList2 = [0] * 18
        moveTypeList2[int(message[19])] = 1
        
        message[20] /= 215
        message[21] /= 100
        message[22] = 1 if message[22] > 0 else 0

        # message[23] = move id2
        # message[24] = move effect id2

        moveTypeList3 = [0] * 18
        moveTypeList3[int(message[25])] = 1
        
        message[26] /= 215
        message[27] /= 100
        message[28] = 1 if message[28] > 0 else 0

        # message[29] = move id2
        # message[30] = move effect id2

        moveTypeList4 = [0] * 18
        moveTypeList4[int(message[31])] = 1
        
        message[32] /= 215
        message[33] /= 100
        message[34] = 1 if message[34] > 0 else 0

        opponentTypeList = [0] * 18
        opponentTypeList[int(message[35])] = 1

        opponentTypeList2 = [0] * 18
        opponentTypeList2[int(message[36])] = 1

        message[37] /= 100
        message[38] /= message[38]
        message[39] /= 714
        message[40] /= 2016
        message[41] /= 2456
        message[42] /= 2016
        message[43] /= 2456
        message[44] /= 2016

      

        return torch.tensor(message[1:])
