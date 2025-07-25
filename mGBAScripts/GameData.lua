--[[ 
Notes:
{rawstring:byte(1, #rawstring)} makes a list of the byte values of the string
emu.memory.cart0 is a reference to the rom
0xAF is the address on the rom that stores the language
0x4A is guess means Japanese
0xFF is a line end
Each name is 11 bytes long but the 11th is a terminator
The _charmap is a table that maps the byte values to characters
We start at speciesmNameTable and move 11 for each id
We read only 10 to avoid terminator
11 bytes is enough for the longest name plus padding/terminator
Takes 11 bytes cause each byte is a binary index for a character

| Hex  | ASCII | Language |
| ---- | ----- | -------- |
| 0x4A | `J`   | Japanese |
| 0x45 | `E`   | English  |
| 0x44 | `D`   | German   |
| 0x46 | `F`   | French   |
| 0x49 | `I`   | Italian  |
| 0x53 | `S`   | Spanish  |

& 0xFFFF includes lower 16 bits cause this directly means only include 16
>> 16 means shift right 16 bits so we only get the upper 16 bits
& 0xFF gets you lower 8 bits
>> 8 means shift right 8 bits so we only get the upper 8 bits
(var >> 8) & 0xFF gets you the middle 8 bits
& 0x1F gets you lower 5 bits
substructure works as each of the 4 structs max size is 12 bytes,
so to make it easier we can break down the 12*8=96 bits into 3 32 bit lists 


I think

ss1[2] & 0xFF gets you the lower 8 bits
(ss1[2] >> 8) & 0xFF gets you the upper 24 bits but then you get the lower 8 of that
(ss1[2] >> 16) & 0xFF gets you the upper 16 bits but then you get the lower 8 of that
ss1[2] >> 24 gets you the upper 8 bits

Scanning the keys returns a hex value that can be a combination of keys,
so for right and start it is 0x18 which is 0x10 and 0x08 which you can combine as (0x10 | 0x08)

This will read from the rom which means static info that doesn't change
emu.memory.cart0:readRange

This will read from the actual emulator memory which is dynamic and changes
emu:readX()

& 0x7F Gets you 7 bits from whatever you are working with as 0x7F in binary is 01111111
& 0xF gets you 4 bits as it is 00001111 in binary
& 0x1 is 1 bit as it is 00000001 in binary


References:
- https://github.com/besteon/Ironmon-Tracker
- https://www.reddit.com/r/pokemonrng/comments/172n3ec/pokemon_gen3_lua_script_for_mgba/
- https://github.com/pret/pokefirered
--]]


-- This is the start of the GameData functionality

-- Represents the ROM and the data on it
local GameData = {
    new = function (self, game)
        self.__index = self
        setmetatable(game, self) 
        return game
    end 
}

MovePath = {}         -- full list of key steps
CurrentMoveIndex = 1  -- index we're on in MovePath
IsAwaitingKeyAck = false

GBA_KEY = {
    A = 0,
    B = 1,
    SELECT = 2,
    START = 3,
    RIGHT = 4,
    LEFT = 5,
    UP = 6,
    DOWN = 7,
    R = 8,
    L = 9
}

-- Character map for converting byte values to characters
GameData.charmap = { [0]=
	" ", "À", "Á", "Â", "Ç", "È", "É", "Ê", "Ë", "Ì", "こ", "Î", "Ï", "Ò", "Ó", "Ô",
	"Œ", "Ù", "Ú", "Û", "Ñ", "ß", "à", "á", "ね", "ç", "è", "é", "ê", "ë", "ì", "ま",
	"î", "ï", "ò", "ó", "ô", "œ", "ù", "ú", "û", "ñ", "º", "ª", "�", "&", "+", "あ",
	"ぃ", "ぅ", "ぇ", "ぉ", "v", "=", "ょ", "が", "ぎ", "ぐ", "げ", "ご", "ざ", "じ", "ず", "ぜ",
	"ぞ", "だ", "ぢ", "づ", "で", "ど", "ば", "び", "ぶ", "べ", "ぼ", "ぱ", "ぴ", "ぷ", "ぺ", "ぽ",
	"っ", "¿", "¡", "P\u{200d}k", "M\u{200d}n", "P\u{200d}o", "K\u{200d}é", "�", "�", "�", "Í", "%", "(", ")", "セ", "ソ",
	"タ", "チ", "ツ", "テ", "ト", "ナ", "ニ", "ヌ", "â", "ノ", "ハ", "ヒ", "フ", "ヘ", "ホ", "í",
	"ミ", "ム", "メ", "モ", "ヤ", "ユ", "ヨ", "ラ", "リ", "⬆", "⬇", "⬅", "➡", "ヲ", "ン", "ァ",
	"ィ", "ゥ", "ェ", "ォ", "ャ", "ュ", "ョ", "ガ", "ギ", "グ", "ゲ", "ゴ", "ザ", "ジ", "ズ", "ゼ",
	"ゾ", "ダ", "ヂ", "ヅ", "デ", "ド", "バ", "ビ", "ブ", "ベ", "ボ", "パ", "ピ", "プ", "ペ", "ポ",
	"ッ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "?", ".", "-", "・",
	"…", "“", "”", "‘", "’", "♂", "♀", "$", ",", "×", "/", "A", "B", "C", "D", "E",
	"F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
	"V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
	"l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "▶",
	":", "Ä", "Ö", "Ü", "ä", "ö", "ü", "⬆", "⬇", "⬅", "�", "�", "�", "�", "�", ""
}


-- Converts a byte string to a human-readable string using the character map
function GameData.toString(game, byteString) 
    local str = ""
    for _, char in ipairs({byteString:byte(1, #byteString)}) do
        if char == 0xFF then
            break
        end
        str = str .. (GameData.charmap[char] or "?")
    end
    return str
end

-- Reads the name of a from the ROM with the given index
function GameData.getPokemonName(game, pokemonIndex)
    local nameAddress = game.romPokemonTable + (pokemonIndex * 11)
    return GameData.toString(game, emu.memory.cart0:readRange(nameAddress, 10))
end

-- Gets the effect id from the given move
function GameData.getMoveEffectID(game, moveID)
    local moveEffectAddress = game.moveData + (moveID * 12)
	return emu.memory.cart0:read8(moveEffectAddress)
end

-- Gets the damage of the given move
function GameData.getMoveDamage(game, moveID)
    local damageAddress = game.moveData + (moveID * 12) + 1
	return emu.memory.cart0:read8(damageAddress)
end

-- Gets the type of the given move
function GameData.getMoveType(game, moveID)
    -- first get the id relating to the type for this move
    local attackTypeID = game.moveData + (moveID * 12) + 2
	local attackTypeNumber = emu.memory.cart0:read8(attackTypeID)
	-- with the type's id, look in the rom's type table to find the type name
	local attackTypeAddress = game.romTypesTable + (attackTypeNumber * 7)
	return emu.memory.cart0:readRange(attackTypeAddress, 6)
end

-- Gets the type ID of the given move
function GameData.getMoveTypeID(game, moveID)
    -- first get the id relating to the type for this move
    local attackTypeID = game.moveData + (moveID * 12) + 2
	return emu.memory.cart0:read8(attackTypeID)
end

-- Gets the accuracy of the given move
function GameData.getMoveAccuracy(game, moveID)
    local accuracyAddress = game.moveData + (moveID * 12) + 3
	return emu.memory.cart0:read8(accuracyAddress)
end

-- Get count of battlers
function GameData.getBattlersCount(game)
    -- This function reads the number of battlers in the current battle
    return emu:read8(Game.battlersCount)
end

function GameData.getDoublesPokemon(game)
    if game:getBattlersCount() ~= 4 then
        return nil
    end

    local pokemon = {
        ["leftOwn"] = nil,
        ["leftOther"] = nil,
        ["rightOwn"] = nil,
        ["rightOther"] = nil
    }
    local leftOwn = emu:read8(game.battlerPartyIndexes) + 1
    pokemon["leftOwn"] = game:getPokemonData(Pokemon[leftOwn][2])

    local leftOther = emu:read8(game.battlerPartyIndexes + 6) + 1
    pokemon["leftOther"] = game:getPokemonData(Pokemon[0][2] + ((leftOther - 1) * 100))
    local rightOwn = emu:read8(game.battlerPartyIndexes + 4) + 1
    pokemon["rightOwn"] = game:getPokemonData(Pokemon[rightOwn][2])

    local rightOther = emu:read8(game.battlerPartyIndexes + 2) + 1
    pokemon["rightOther"] = game:getPokemonData(Pokemon[0][2] + ((rightOther - 1) * 100))
    return pokemon
end

--[[
Gets the target of the given move
Values are:
---------------------------------
0x00	Selected target         |
0x01	Depends on the attack   |
0x02	Unused                  |
0x04	Random target           |
0x08	Both foes               |
0x10	User                    |
0x20	Both foes and partner   |
0x40	Opponent field          |
---------------------------------
--]]
function GameData.getMoveTarget(game, moveID)
    local damageAddress = game.moveData + (moveID * 12) + 6
	return emu.memory.cart0:read8(damageAddress)
end

-- Reads the address of a Pokemon in the player's party and collects the pokemon data
function GameData.getPokemonData(game, pokemonAddress)
    local pokemon = {}
    pokemon.status = emu:read32(pokemonAddress + 80)
	pokemon.level = emu:read8(pokemonAddress + 84)
	pokemon.hp = emu:read16(pokemonAddress + 86)
    pokemon.stats = {
        -- HP stat
		emu:read16(pokemonAddress + 88),
        -- Attack stat
		emu:read16(pokemonAddress + 90),
        -- Defense stat
		emu:read16(pokemonAddress + 92),
        -- Special Attack stat
		emu:read16(pokemonAddress + 96),
        -- Special Defense stat
		emu:read16(pokemonAddress + 98),
        -- Speed stat
		emu:read16(pokemonAddress + 94)
	}

    pokemon.nickname = game:toString(emu:readRange(pokemonAddress + 8, 10))
    pokemon.personality = emu:read32(pokemonAddress + 0)
    pokemon.otId = emu:read32(pokemonAddress + 4)

    -- Perumations and ordering of the 4 structs that contain some pokemon data
    local key = pokemon.otId ~ pokemon.personality
    local substructPermutations= {
		[ 0] = {0, 1, 2, 3},
		[ 1] = {0, 1, 3, 2},
		[ 2] = {0, 2, 1, 3},
		[ 3] = {0, 3, 1, 2},
		[ 4] = {0, 2, 3, 1},
		[ 5] = {0, 3, 2, 1},
		[ 6] = {1, 0, 2, 3},
		[ 7] = {1, 0, 3, 2},
		[ 8] = {2, 0, 1, 3},
		[ 9] = {3, 0, 1, 2},
		[10] = {2, 0, 3, 1},
		[11] = {3, 0, 2, 1},
		[12] = {1, 2, 0, 3},
		[13] = {1, 3, 0, 2},
		[14] = {2, 1, 0, 3},
		[15] = {3, 1, 0, 2},
		[16] = {2, 3, 0, 1},
		[17] = {3, 2, 0, 1},
		[18] = {1, 2, 3, 0},
		[19] = {1, 3, 2, 0},
		[20] = {2, 1, 3, 0},
		[21] = {3, 1, 2, 0},
		[22] = {2, 3, 1, 0},
		[23] = {3, 2, 1, 0},
	}

    local pSel = substructPermutations[pokemon.personality % 24]
	local substruct0 = {}
	local substruct1 = {}
	local substruct2 = {}
	local substruct3 = {}

    -- Reading the 12 bytes/96 bits of each substruct
    -- store 4 bytes/32 bits in each substruct
    for i = 0, 2 do
		substruct0[i] = emu:read32(pokemonAddress + 32 + pSel[1] * 12 + i * 4) ~ key
		substruct1[i] = emu:read32(pokemonAddress + 32 + pSel[2] * 12 + i * 4) ~ key
		substruct2[i] = emu:read32(pokemonAddress + 32 + pSel[3] * 12 + i * 4) ~ key
		substruct3[i] = emu:read32(pokemonAddress + 32 + pSel[4] * 12 + i * 4) ~ key
	end

    -- Get the species from the first substruct and only get the lower 16 bits of the first 32
    -- bits of the first substruct
    pokemon.species = substruct0[0] & 0xFFFF

    -- 0x1c is the size of base stats?
    local addrOffset = game.speciesInfo + (pokemon.species * 0x1C)

    pokemon.type1 = emu:read8(addrOffset + 6)
    pokemon.type2 = emu:read8(addrOffset + 7)

    -- Get the index of each of the 4 moves of the pokemon
    -- moves take up 64 bits of the second substruct
    pokemon.moves = {
        -- get the first lower 16 bits of the first 32 bits of the second substruct
		substruct1[0] & 0xFFFF,
        -- get the first upper 16 bits of the first 32 bits of the second substruct
        -- by bit shifting the first 32 bits of the second substruct to the right by 16
		substruct1[0] >> 16,
        -- get the second lower 16 bits of the first 32 bits of the second substruct
		substruct1[1] & 0xFFFF,
        -- get the second upper 16 bits of the first 32 bits of the second substruct
        -- by bit shifting the first 32 bits of the second substruct to the right by 16
		substruct1[1] >> 16
	}

    pokemon.movesDamage = {}
    for i = 1, 4 do
        local moveID = pokemon.moves[i]
        local damage = game:getMoveDamage(moveID)

        table.insert(pokemon.movesDamage, damage)
    end

    pokemon.movesType = {}
    for i = 1, 4 do
        local moveID = pokemon.moves[i]
        local moveType = game:getMoveType(moveID)

        table.insert(pokemon.movesType, moveType)
    end 

    pokemon.movesAccuracy = {}
    for i = 1, 4 do
        local moveID = pokemon.moves[i]
        local moveAccuracy = game:getMoveAccuracy(moveID)

        table.insert(pokemon.movesAccuracy, moveAccuracy)
    end


    

    -- get the remaining pp for each of the 4 moves
    -- moves take up the last 32 bits of the second substruct
    pokemon.pp = {
        -- get the first lower 8 bits in the 32 bit space
		substruct1[2] & 0xFF,
        -- shift 8 to the right to get the upper 24 bits 
        -- and then only get the lower 8 bits of that
		(substruct1[2] >> 8) & 0xFF,
        -- shift 16 to the right to get the upper 16 bits 
        -- and then only get the lower 8 bits of that
		(substruct1[2] >> 16) & 0xFF,
        -- shift 24 to the right to get the remaining upper 8 bits 
		substruct1[2] >> 24
	}

    -- Get the EVs of the pokemon from the third substruct
    -- Moves take up 48 bits of the third substruct's 96 bits
    -- the first 4 evs take up the full first 32 bits and the last 2 evs
    -- take up the first 16 bits of the second 32 bits
    pokemon.EVs = {
        -- get the first lower 8 bits in the 32 bit space (hp)
		substruct2[0] & 0xFF,
        -- shift 8 to the right to get the upper 24 bits 
        -- and then only get the lower 8 bits of that (attack)
		(substruct2[0] >> 8) & 0xFF,
        -- shift 16 to the right to get the upper 16 bits 
        -- and then only get the lower 8 bits of that (defense)
		(substruct2[0] >> 16) & 0xFF,
        -- shift 24 to the right to get the remaining upper 8 bits (speed)
        substruct2[0] >> 24,
        -- now in the next 32 bits, get the first lower 8 bits (special attack)
		substruct2[1] & 0xFF,
        -- shift 8 to the right to get the upper 24 bits 
        -- and then only get the lower 8 bits of that (special defense)
		(substruct2[1] >> 8) & 0xFF,
	}

    ability1 = emu:read8(addrOffset + 22)
    ability2 = emu:read8(addrOffset + 23)
    abilityID = (substruct3[1] >> 31) & 1
    pokemon.ability = (abilityID == 0) and ability1 or ability2

    return pokemon
end

function SendButtonPress(button) 
    console:log("in send button")
    if button == nil then
        console:error("sendButtonPress: 'button' is nil!")
        return
    end
    emu:addKey(button)
    --emu:runFrame()
    --emu:clearKey(button)
end

function GameData.moveCursor(game,targetMove)
    console:log("In move cursor")
    local currentMove = game:getCursorSelection()

    local moveMap = {
        [0] = { [0] = {0, 0}, [1] = {0, 4, 0}, [2] = {0, 7, 0}, [3] = {0, 4, 7, 0} },
        [1] = { [0] = {0, 5, 0}, [1] = {0, 0}, [2] = {0, 5, 7, 0}, [3] = {0, 4, 0} },
        [2] = { [0] = {0, 6, 0}, [1] = {0, 4, 6, 0}, [2] = {0, 0}, [3] = {0, 4, 0} },
        [3] = { [0] = {0, 5, 6, 0}, [1] = {0, 6, 0}, [2] = {0, 5, 0}, [3] = {0, 0} },
        [4] = { [-1] = {7, 0, 4, 7, 0, 0}},
        [5] = { [-1] = {7, 0, 4, 7, 7, 0, 0}},
        [6] = { [-1] = {7, 0, 4, 7, 7, 7, 0, 0}},
        [7] = { [-1] = {7, 0, 4, 7, 7, 7, 7, 0, 0}},
        [8] = { [-1] = {7, 0, 4, 7, 7, 7, 7, 7, 0, 0}}
    }

    if currentMove == -6 then
        console:log("Unknown move")
        return
    end

    local path 

    if targetMove >= 4 and targetMove <= 8 then
        path = moveMap[targetMove][-1]
    else
        path = moveMap[currentMove][targetMove]
    end
    

    return path
end



--[[
This function reads the battle address to determine if a battle is ongoing
_____________________
0   |   In battle   |
1   |   Won         |
2   |   Lost        |
4   |   Fled        |
7   |   Caught      |
---------------------
--]]
function readBattleAddress()
    return emu:read8(Game.battleAddress)
end

-- Get the turn count of the current battle
function getTurnCount()
    return emu:read8(Game.turnCount)
end

-- Get the last used move ID in the current battle
function getLastUsedMoveID()
    return emu:read8(Game.lastUsedMove)
end


--[[
Sends a message to the python socket with the current turn data
if we are on training mode it sends the old turn data and the move made for that turn
otherwise it sends the current turn data and requests a move
Sends data in the form of:
____________________________________________________________________________
Player Pokemon      |   type, type2, level, ability, status,               |
                    |   currenthp, hp, atk, def, spatk, spdef,             |
                    |   spd, moveXID, moveEffectID,moveXType, moveXDamage, |
                    |   moveXAccuracy, moveXpp                             |
                    |                                                      |
Opponent Pokemon    |   type, type2, level, ability, status, currenthp,    |
                    |   hp, atk, def, spatk, spdef, spd                    |
Party Pokemon (2-6) |   type, type2, level, ability, status, currenthp,    |
                    |   hp, atk, def, spatk, spdef, spd, switchable        |
----------------------------------------------------------------------------
--]]
function GameData.contactPythonSocket(game, currentPokemon)
    console:log("Preparing to message python...")

    local command = "REQUEST_AI_MOVE"
    local turnData = game:getTurnData(currentPokemon)

    if TrainingMode then
        command = "SAVE_MOVE"
        turnData = TurnData
    end

    local battleData = {command, table.unpack(turnData)}

    -- Add the data for each party pokemon
    for i = 2, 6 do
        local partyPokemon = game:getPokemonData(Pokemon[i][2])
        local formattedPartyPokemon = game:formatPlayerPartyPokemon(partyPokemon)
        for _, v in ipairs(formattedPartyPokemon) do
            table.insert(battleData, v)
        end
    end

    if TrainingMode then
        -- Add the choice that the player made for the turn
        local playerDecision = game:getTurnDecision(currentPokemon)
        table.insert(battleData, playerDecision)
    else
        table.insert(battleData, -1)
    end

    for i, v in ipairs(battleData) do
        if v == nil then
            console:error(string.format("Data[%d] is nil", i))
        elseif v == -1 then
            console:error(string.format("Data[%d] is -1", i))
        end
    end


    -- turn the table into a string
    local pokemonData = tableAsString(battleData)

    for i, v in ipairs(battleData) do
        --console:log(string.format("Data[%d] = %s (type: %s)", i, tostring(v), type(v)))
    end

    console:log("Sending data: " .. pokemonData)
    SendMessageToServer(pokemonData)
    console:log("Finished sending data.")
end

--[[
Sends a message to the python socket of the turn data and choice the player has made
____________________________________________________________________________
Player Pokemon      |   type, type2, level, status, currenthp, hp,         |
                    |   atk, def, spatk, spdef, spd, moveXID, moveEffectID,|
                    |   moveXType, moveXDamage, moveXAccuracy, moveXpp     |
                    |                                                      |
Opponent Pokemon    |   type, type2, level, status, currenthp, hp, atk,    |
                    |   def, spatk, spdef, spd                             |
Party Pokemon (2-6) |   type, type2, level, status, currenthp, hp, atk,    |
                    |    def, spatk, spdef, spd, switchable                |
Chosen Move/Switch  |   0-8                                                |
----------------------------------------------------------------------------
--]]
--[[
function GameData.sendTrainingData(game, currentPokemon)
    console:log("Sending to Python the turn to be saved...")

    -- Start the message with the command and the turn data
    local command = "SAVE_MOVE"
    local battleData = {command, table.unpack(TurnData)}

    -- Add the data for each party pokemon
    for i = 2, 6 do
        local partyPokemon = game:getPokemonData(Pokemon[i][2])
        local formattedPartyPokemon = game:formatPlayerPartyPokemon(partyPokemon)
        for _, v in ipairs(formattedPartyPokemon) do
            table.insert(battleData, v)
        end
    end
    -- Add the choice that the player made for the turn
    local playerDecision = game:getTurnDecision(currentPokemon)
    table.insert(battleData, playerDecision)

    -- turn the table into a string
    local pokemonData = tableAsString(battleData)

    console:log("Sending turn data: " .. pokemonData)
    SendMessageToServer(pokemonData)
    console:log("Finished sending turn data.")
end
--]]

--[[
This function gets the decision made by the player in the current turn
0-3 represents the move index chosen by the player
4-8 represents the switch index chosen by the player (To get the actual index subtract 2)
--]]
function GameData.getTurnDecision(game, currentPokemon)
    console:log("Getting the player's decision for the current turn...")
    -- first lets check for a move choice
    local decision = -1
    local chosenMoveIndex = -1
    for i = 1, 4 do
        local moveData = game:moveAsList(currentPokemon, i)
        if moveData[1] == getLastUsedMoveID() then
            chosenMoveIndex = i - 1
        end
    end

    console:log(string.format("Chosen move index is: %d \n Last Used MoveID is %i", chosenMoveIndex, getLastUsedMoveID()))
    -- if the chosen move index is -1, it means the player did not choose a move so lets see
    -- if they switched to another pokemon
    if chosenMoveIndex == -1 then
        decision = game.getSwitchChoice(game, currentPokemon)
    else
        console:log("Chosen move index is: " .. chosenMoveIndex)
        decision = chosenMoveIndex
        --game.moveCursor(decision, getLastUsedMoveID())
    end

    if decision == -1 then
        console:error("No move or switch was found for the player in the current turn!")
    end
    return decision
end


-- If the player did not choose a move, find which pokemon they switched to
function GameData.getSwitchChoice(game, currentPokemon)
    console:log("Chosen move index is -1, this means the last used move was not found in the player's moves! or maybe it means switched")
    -- now lets check for a switch
    local leftOwn = emu:read8(game.battlerPartyIndexes) + 1
    local newActivePokemon = game:getPokemonData(Pokemon[leftOwn][2])
    console:log(string.format("Left own is : %d, current pokemon species is: %d", leftOwn, currentPokemon.species))
    local oldActivePokemon = LastActivePokemon
    console:log("Current Pokemon: " .. game:getPokemonName(newActivePokemon.species))
    console:log("Old Pokemon: " .. game:getPokemonName(oldActivePokemon.species))
    if newActivePokemon ~= oldActivePokemon then
        console:log("Pokemon switched from " .. game:getPokemonName(oldActivePokemon.species) .. " to " .. game:getPokemonName(newActivePokemon.species))
    else
        console:error("Current Pokemon is the same as the old Pokemon, this should not happen!")
    end
    -- we return index + 2 to indicate a switch as part of the player decision field
    console:log(string.format("Switched to index %d which is treated as %d", leftOwn, leftOwn + 3))
    return leftOwn + 2
end

-- get the current turn's data,
-- which currently is the player's pokemon data and the opponent's pokemon data
function GameData.getTurnData(game, currentPokemon)
    local playerPokemonData = game:formatActivePokemon(currentPokemon, true)

    local opponentPokemon = game:getPokemonData(Game.enemyParty)
    local opponentPokemonData = game:formatActivePokemon(opponentPokemon, false)

    -- combine tables
    for _,data in ipairs(opponentPokemonData) do
        table.insert(playerPokemonData, data)
    end

    return playerPokemonData
end


--[[
Format's the pokemon data for the player and opponent in a format that can be sent to the server
to Request the Neural Network for what choice the player should make
function GameData.requestAIMove(game, currentPokemon)
    console:log("Requesting AI move for current Pokemon...")

    local command = "REQUEST_AI_MOVE,"
    local currentTurnData = game:getTurnData(currentPokemon)

    local battleData = {command, table.unpack(currentTurnData)}

    -- Add the data for each party pokemon
    for i = 2, 6 do
        local partyPokemon = game:getPokemonData(Pokemon[i][2])
        local formattedPartyPokemon = game:formatPlayerPartyPokemon(partyPokemon)
        for _, v in ipairs(formattedPartyPokemon) do
            table.insert(battleData, v)
        end
    end

    -- turn the table into a string
    local pokemonData = tableAsString(battleData)

    console:log("Sending AI move request with data: " .. command)
    SendMessageToServer(command)
    console:log("Finished sending AI move request.")
end
--]]

-- Formats a given pokemon into the a form that lists its data like types, level, status, level, etc
-- if the pokemon is not the active pokemon out then we also have to add an extra field to the end
-- that clarifies if the pokemon is switchable or not
function GameData.formatPlayerPartyPokemon(game, playerPokemon)
    -- Get all the player pokemon's data in the format of:
    -- type, type2, level, ability, status, currnethp, hp, atk, def, spatk, spdef, spd, moveXID, moveEffectID, moveXType, moveXDamage, moveXAccuracy, moveXpp
    local playerPokemonData = {}

    -- if not active we need to first check if the pokemon is alive or not
    -- if they are then we send type1, type2, level, ability, status, currenthp, hp, atk, def, spatk, spdef, spd, 1
    -- if they are not then we send 0 for all stats
    if playerPokemon.species == 0 then
        console:log("Pokemon is null")
        for i = 1, 13 do
            table.insert(playerPokemonData, 0)
        end
    else
        table.insert(playerPokemonData, playerPokemon.type1)
        table.insert(playerPokemonData, playerPokemon.type2)
        table.insert(playerPokemonData, playerPokemon.level)
        table.insert(playerPokemonData, playerPokemon.ability)
        table.insert(playerPokemonData, playerPokemon.status)
        table.insert(playerPokemonData, playerPokemon.hp)
        -- If not active, just add the base stats
        for i = 1,6 do
            table.insert(playerPokemonData, playerPokemon.stats[i])
        end
        table.insert(playerPokemonData, 1) -- 1 means the pokemon is alive and switchable
    end

    return playerPokemonData
end

function GameData.formatActivePokemon(game, pokemon, playerPokemon)
    local pokemonData = {}

    -- add the stats
    table.insert(pokemonData, pokemon.type1)
    table.insert(pokemonData, pokemon.type2)
    table.insert(pokemonData, pokemon.level)
    table.insert(pokemonData, pokemon.ability)
    table.insert(pokemonData, pokemon.status)
    table.insert(pokemonData, pokemon.hp)

    local statStageAddress = Game.opponentBattlePokemonStruct + 24
    if playerPokemon then
        statStageAddress = Game.playerBattlePokemonStruct + 24
    end

    -- get the pokemon's stat stages (treated as -6-+6) that affect the stats
    console:log("Reading stat stages from address: " .. statStageAddress)
    local statStages = {
        -- hp
        emu:read8(statStageAddress),
        -- attack
        emu:read8(statStageAddress + 1),
        -- defense
        emu:read8(statStageAddress + 2),
        -- Sp.Atk
        emu:read8(statStageAddress + 4),
        -- Sp.Def
        emu:read8(statStageAddress + 5),
        -- Spd
        emu:read8(statStageAddress + 3),
        -- Accuracy
        -- Evasion
    }

    -- Add the stats of the player pokemon
    for i = 1,6 do
        table.insert(pokemonData, getEffectiveStat(pokemon.stats[i], statStages[i] - 6))
    end

    -- if its the active player pokemon we want to get the moves
    if playerPokemon then
        local chosenMoveIndex = -1
        local moves = {}

        for i = 1, 4 do
            local moveData = game:moveAsList(pokemon, i)
            for _, v in ipairs(moveData) do
                table.insert(moves, v)
            end
        end

        -- Add the moves of the player pokemon
        for _,moveData in ipairs(moves) do
            table.insert(pokemonData, moveData)
        end
    end

    return pokemonData
end


--
function GameData.formatOpponentPokemon(game, opponentPokemon)
    local opposingStatStageAddress = Game.opponentBattlePokemonStruct + 24
    local opponentPokemonStatStages = {
        -- hp
        emu:read8(opposingStatStageAddress),
        -- attack
        emu:read8(opposingStatStageAddress + 1),
        -- defense
        emu:read8(opposingStatStageAddress + 2),
        -- Sp.Atk
        emu:read8(opposingStatStageAddress + 4),
        -- Sp.Def
        emu:read8(opposingStatStageAddress + 5),
        -- Spd
        emu:read8(opposingStatStageAddress + 3),
        -- Accuracy
        -- Evasion
    }

    -- format the opponent pokemon as:
    -- type, type2, level, ability, status, currenthp, hp, atk, def, spatk, spdef, spd
    local opponentPokemonData = {
        opponentPokemon.type1,
        opponentPokemon.type2,
        opponentPokemon.level,
        opponentPokemon.ability,
        opponentPokemon.status,
        opponentPokemon.hp
    }

     -- Add the stats of the opposing pokemon
    for i = 1,6 do
        table.insert(opponentPokemonData,
        getEffectiveStat(opponentPokemon.stats[i], opponentPokemonStatStages[i] - 6))
        console:log(string.format("Opponent Pokemon Stat %s %d: %f, Stat Stage: %f", stringStats[i], i, opponentPokemon.stats[i], opponentPokemonStatStages[i] - 6))
    end

    return opponentPokemonData
end

-- This function returns the move data as a list
-- index refers to the move id that is used to index the move data
function GameData.moveAsList(game, pokemon, index)
    -- This function returns the move data as a list
    local moveID = pokemon.moves[index]
    local moveData = {
        moveID,
        game:getMoveEffectID(moveID),
        game:getMoveTypeID(moveID),
        pokemon.movesDamage[index],
        pokemon.movesAccuracy[index],
        pokemon.pp[index]
    }
    console:log(string.format("Move %i: %s", index, table.concat(moveData, ", ")))
    return moveData
end

--[[
Not the best way to check if the cursor is in the pokemon party, bag, or a move
-- When 0x200E728 changes 38 means in pokemon party and 14 is bag
        -- Topleft:     0x200F4F6 = 0x01,  0x200F508 = 0x20, 0x200F536 = 0x02, 0x200F548 = 0x20, 0x200F576 = 0x20, 0x200F588 = 0x20, 0x200F5B6 = 0x20, 0x200F5C8 = 0x20
        -- Bottomleft:  0x200F4F6 = 0x20,  0x200F508 = 0x20, 0x200F536 = 0x20, 0x200F548 = 0x20, 0x200F576 = 0x01, 0x200F588 = 0x20, 0x200F5B6 = 0x02, 0x200F5C8 = 0x20
        -- TopRight:    0x200F4F6 = 0x20,  0x200F508 = 0x01, 0x200F536 = 0x20, 0x200F548 = 0x02, 0x200F576 = 0x20, 0x200F588 = 0x20, 0x200F5B6 = 0x20, 0x200F5C8 = 0x20
        -- BottomRight: 0x200F4F6 = 0x20,  0x200F508 = 0x20, 0x200F536 = 0x20, 0x200F548 = 0x20, 0x200F576 = 0x20, 0x200F588 = 0x01, 0x200F5B6 = 0x20, 0x200F5C8 = 0x02
--]]
function GameData.getCursorSelection(game)
    if readBattleAddress() ~= 0 then
        return -1
    end

    local topLeft = {
        addresses =  {0x200F4F6, 0x200f536},
        values = {0x01, 0x02}
    }

    local topRight = {
        addresses = {0x200F508, 0X200F548},
        values = {0x01, 0x02}
    }

    local bottomLeft = {
        addresses = {0x200F576, 0x200F5B6},
        values = {0x01, 0x02}
    }

    local bottomRight = {
        addresses = {0x200F588, 0x200F5C8},
        values = {0x01, 0x02}
    }

    local function locateCursor(addresses, expectedPositions)
        for i = 1, #addresses do
            if emu:read8(addresses[i]) ~= expectedPositions[i] then
                -- console:log("printed false")
                return false
            end
        end
        return true
    end

    local inPartyOrBag = emu:read8(0x200E728)
    --console:log("Cursor is in: " .. string.format("%i", inPartyOrBag))

    if inPartyOrBag == 0x38 then
        -- console:log("Cursor is in the Pokemon Party")
        return 4
    elseif inPartyOrBag == 0x14 then
        --console:log("Cursor is in the Bag")
        return 5
    else
        --console:log("Cursor is not in the Pokemon Party or Bag, it is in the Moves")

        if locateCursor(topLeft.addresses, topLeft.values) then
            --console:log("Cursor is in the Top Left Move")
            return 0
        elseif locateCursor(topRight.addresses, topRight.values) then
            --console:log("Cursor is in the Top Right Move")
            return 1
        elseif locateCursor(bottomLeft.addresses, bottomLeft.values) then 
            --console:log("Cursor is in the Bottom Left Move")
            return 2
        elseif locateCursor(bottomRight.addresses, bottomRight.values) then
            --console:log("Cursor is in the Bottom Right Move")
            return 3
        end
        --console:log("Player is in the battle's menu")
        return 6
    end
end

-- Checks if the player is in the Safari Zone
function GameData.checkIfInSafariZone(game)
    local saveBlock = emu:read32(game.saveBlockPtr)
    local safariOffset = 0x800 + 0x0
    local flagOffset = 0x0EE0
    local safariZoneAddress = saveBlock + flagOffset + math.floor(safariOffset / 8)
    local safariBit = safariOffset % 8
    local safariZoneFlag = emu:read8(safariZoneAddress)
    return math.floor((safariZoneFlag >> safariBit) % 2) ~= 0
end

-- Gets the effective stat of a pokemon by using the stat stage
function getEffectiveStat(base, stage)
    if base == nil or stage == nil then
        console:error("Base or stage is nil in getEffectiveStat!")
        return 0
    end
    local stageMultipliers = {
        [-6] = 2/8, [-5] = 2/7, [-4] = 2/6, [-3] = 2/5, [-2] = 2/4, [-1] = 2/3,
        [0]  = 1.0,
        [1]  = 3/2, [2] = 4/2, [3] = 5/2, [4] = 6/2, [5] = 7/2, [6] = 8/2,
    }
    return base * stageMultipliers[stage]
end

function getEffectiveAccuracyEvasionStat(base, stage)
    if base == nil or stage == nil then
        console:error("Base or stage is nil in getEffectiveStat!")
        return 0
    end
    local stageMultipliers = {
        [-6] = 2/8, [-5] = 2/7, [-4] = 2/6, [-3] = 2/5, [-2] = 2/4, [-1] = 2/3,
        [0]  = 1.0,
        [1]  = 3/2, [2] = 4/2, [3] = 5/2, [4] = 6/2, [5] = 7/2, [6] = 8/2,
    }
    return base * stageMultipliers[stage]
end



-- Print the data from a given table
function tableAsString(data)
    console:log("Converting table to string...")
    if data == nil or #data == 0 then
        console:error("Data is nil or empty!")
        return ""
    end
    local stringFormat = ""

    for _,data in ipairs(data) do
        stringFormat = stringFormat .. "%s,"
    end

    -- Remove the last comma
    stringFormat = string.sub(stringFormat, 1, -2)


    local stringData = string.format(
        stringFormat,
        table.unpack(data)
    )
    return stringData
end


-- This is the start of functionality associated the the mGBA emulator

function InitializeGame()
    -- Represents the FireRed rom data
    Game = GameData:new({
        name = "FireRed (USA)",

        -- Address for where the first pokemon in the player's party is stored
        playerParty = 0x2024284,
        -- Address that stores the count of pokemon in the player's party (between 1 - 6)
        partyCount = 0x2024029,
        -- Address for the rom's table of the pokemon names
        romPokemonTable = 0x245F50,
        -- Address for the enemy pokemon data 
        -- (could be a wild pokemon or the first pokemon of a enemy trainer)
        enemyParty = 0x0202402C,
        -- Address for my best guess at what signifies a battle
        battleAddress = 0x2023E8A,
        -- 0x020386B4
        -- Address for where the rom stores all the names of the moves
        moveNames = (0x00247110) - 13,
        -- Address for where the rom stores the move data
        moveData = (0x00250C80 - 12),
        -- Address for where the rom stores the pokemon types
        romTypesTable = (0x0024F210),
        -- Has all the info about the pokemon species like types, and base stats
        speciesInfo = 0x82547F4,
        -- Address that holds the count of battlers?
        battlersCount = 0x2023BCC,
        -- Address representing the battle struct for the player's pokemon
        playerBattlePokemonStruct = 0x2023BE4,
        -- Address representing the battle struct for the opposing pokemon
        opponentBattlePokemonStruct = 0x2023C3C,
        -- Address for the battle results
        battleResults = 0x3004F90,
        -- Address for the current battle's turn count
        turnCount = 0x3004FA3,
        -- Address for the last used move in the current battle
        lastUsedMove = 0x3004FB2,
        -- Address for the indexs of the 4 pokemon in a double battle
        battlerPartyIndexes = 0x2023BCE,
        saveBlockPtr = 0x3005008

    })
    if not Game then
        console:error("Failed to initialize game data!")
        return
    end

    PrintBuffer = console:createBuffer("Print")
    Frame = 0
    Pokemon = {
			[0] = {"WILD/ENEMY POKEMON", Game.enemyParty},
			[1] = {"POKEMON 1", Game.playerParty},
			[2] = {"POKEMON 2", Game.playerParty + 100},
			[3] = {"POKEMON 3", Game.playerParty + 200},
			[4] = {"POKEMON 4", Game.playerParty + 300},
			[5] = {"POKEMON 5", Game.playerParty + 400},
			[6] = {"POKEMON 6", Game.playerParty + 500},
		}
    CurrentSelectedPokemon = 1
    CurrentMoveIndex = 0
    CurrentTurn = 0
    LastPressedKey = nil
    initializeSocketConnection()
    UseBattleAI = false
    SocketCommunicating = false
    local leftOwn = emu:read8(Game.battlerPartyIndexes) + 1
    LastActivePokemon = Game:getPokemonData(Pokemon[leftOwn][2])
    TurnData = {}
    TrainingMode = false
    InCombat = false
    InBattle = readBattleAddress()
    CurrentBattleMenuSelect = -1
    console:log("Game initialized successfully!")
end


HEX_KEYS = {
    RIGHT = 0x10,
    LEFT = 0x20,
    UP = 0x40,
    DOWN = 0x80,
    START = 0x08,
    A_X = 0x01,
    B_Z = 0x02,
}

function Input()
    local selectedKey = emu:getKeys()
    local partyCount = emu:read8(Game.partyCount)
    if selectedKey ~= LastPressedKey then
        -- console:log(string.format("Current Selected Key: 0x%02X", selectedKey))
        LastPressedKey = selectedKey
        if (LastPressedKey == (HEX_KEYS.A_X | HEX_KEYS.RIGHT) and readBattleAddress() == 0) then
            local battlersCount = Game:getBattlersCount()
            if battlersCount == 4 then
                console:log("Cannot currently use AI in a double battle!")
            elseif battlersCount > 1 then
                console:log("Pressed Activate AI")
                UseBattleAI = true
                if TrainingMode then
                    console:log("Training Mode is active so we are deactivating it for AI")
                    TrainingMode = false
                end
                local leftOwn = emu:read8(Game.battlerPartyIndexes) + 1
                LastActivePokemon = Game:getPokemonData(Pokemon[leftOwn][2])
            else
                console:log("Cannot use AI in a single or safari battle!")
            end
        elseif LastPressedKey == (HEX_KEYS.B_Z | HEX_KEYS.LEFT) then
            if SocketCommunicating then
                console:log("Battle AI is thinking! Unable to cancel while in progress.")
            else
                console:log("Pressed Deactivate AI")
                UseBattleAI = false
            end
        elseif LastPressedKey == (HEX_KEYS.UP | HEX_KEYS.A_X) then
            if TrainingMode then
                console:log("Training Mode is already active!")
            else
                console:log("Pressed Activate Training Mode")
                TrainingMode = true
                if useBattleAI then
                    console:log("Battle AI is active so we are deactivating it for TrainingMode")
                    useBattleAI = false
                end
            end
        elseif LastPressedKey == (HEX_KEYS.DOWN | HEX_KEYS.A_X) then
            if not TrainingMode then
                console:log("Training Mode is already deactivated!")
            else
                console:log("Pressed Deactivate Training Mode")
                TrainingMode = false
            end
        end
    end
end

function Update()
    if not Game or not PrintBuffer then
        PrintBuffer:print("Game or PrintBuffer is not initialized!")
        return
    end
    

    CurrentPokemon = Pokemon[CurrentSelectedPokemon][2]
    local leftOwn = emu:read8(Game.battlerPartyIndexes) + 1
    currentActivePlayerPokemon = Game:getPokemonData(Pokemon[leftOwn][2])

    if not CurrentPokemon then
        PrintBuffer:print("Current Pokemon address is nil!")
        return
    end
    
    if not emu:read32(CurrentPokemon) then
        PrintBuffer:print("Current Pokemon address is invalid!")
        return
    end

    if not Game:getPokemonData(CurrentPokemon) then
        PrintBuffer:print("Failed to get Pokemon data!")
        return
    end

    if not readBattleAddress() then
        PrintBuffer:print("GameData battle address is invalid!")
        return
    end

    if InBattle ~= readBattleAddress() and readBattleAddress() == 0 and not Game:checkIfInSafariZone() then
        console:log("First entering battle!")
        console:log(string.format("Battler count: %d, Our pokemon species %s", Game:getBattlersCount(), emu:read16(Game.playerBattlePokemonStruct)))
        console:log(string.format("If in Safari: %s", Game:checkIfInSafariZone()))
        local leftOwn = emu:read8(Game.battlerPartyIndexes) + 1
        local leftOwnPoke = Game:getPokemonData(Pokemon[leftOwn][2])
        console:log(string.format("Left Own Pokemon: %s, Species: %d", Game:getPokemonName(leftOwnPoke.species), leftOwnPoke.species))

        local rightOther = emu:read8(Game.battlerPartyIndexes + 2) + 1
        local rightOtherPokemon = Game:getPokemonData(Pokemon[0][2] + ((rightOther - 1) * 100))
        console:log(string.format("Right Other Pokemon: %s, Species: %d", Game:getPokemonName(rightOtherPokemon.species), rightOtherPokemon.species))

        InBattle = readBattleAddress()
        TurnData = Game:getTurnData(currentActivePlayerPokemon)
        CurrentBattleMenuSelect = Game:getCursorSelection()
    end

    -- if readBattleAddress() == 0 and Game:getCursorSelection() ~= CurrentBattleMenuSelect then
    --     console:log("Battle Menu Selection Changed!")
    --     console:log(string.format("Current Battle Menu Selection: %d, Previous Battle Menu Selection: %d", Game:getCursorSelection(), CurrentBattleMenuSelect))
    --     CurrentBattleMenuSelect = Game:getCursorSelection()
    -- end



    --[[
    Check if the turn has changed and we're currently in a singles battle
    If the turn has changed, depending on the mode we are in
    useBattleAI: we send the data of the current turn and request a move from the Neural Network
    TrainingMode: we send the data of the last turn and the move made during it and save it in python
    At the end we need to update the turn data and the active pokemon of the turn
    --]]
    if CurrentTurn ~= getTurnCount() and readBattleAddress() == 0 and Game:getBattlersCount() == 2 and not Game:checkIfInSafariZone() then
        console:log(string.format("Turn Changed, Current Turn: %i, Previous Turn: %i, Last Used Move ID: %i", getTurnCount(), CurrentTurn, getLastUsedMoveID()))
        CurrentTurn = getTurnCount()
        if FirstTurn then
            console:log("Turn Changed so we update FirstTurn")
            FirstTurn = false
        end

        if UseBattleAI then
            if not SocketCommunicating then
                console:log("Sending turn data...")
                SocketCommunicating = true
                Game:contactPythonSocket(currentActivePlayerPokemon)
                -- Game:requestAIMove(currentActivePlayerPokemon, true)
                console:log("Turn Data sent!")
                -- On message receive turn off SocketCommunicating
            end
        end

        -- If we are in training mode, we send the turn data to the server
        if TrainingMode and not SocketCommunicating then
            console:log("Sending turn   data for training...")
            SocketCommunicating = true
            Game:contactPythonSocket(currentActivePlayerPokemon)
            console:log("Turn Data sent for training!")
        end
        -- Save the turn data
        TurnData = Game:getTurnData(currentActivePlayerPokemon)
        -- Save the current Active Pokemon
        LastActivePokemon = currentActivePlayerPokemon
    end

    -- Make sure to still send last turn data if we are in training mode
    -- Basically just for if a battle ends before the end of the first turn
    -- and for the last turn of the battle
    if InBattle ~= readBattleAddress() and readBattleAddress() ~= 0 and Game:getBattlersCount() == 2 and not Game:checkIfInSafariZone() then
        console:log("Battle Ended in the first turn, so we need to send if necessary")
        FirstTurn = false
        CurrentBattleMenuSelect = -1
        -- If we are in training mode, we send the turn data to the server
        if TrainingMode and not SocketCommunicating then
            console:log("Sending turn data for training...")
            SocketCommunicating = true
            Game:contactPythonSocket(currentActivePlayerPokemon)
            console:log("Turn Data sent for training!")
        end
    end

    -- need to check if we entered battle and if we are still in battle and when things/turn changes
    if Prev==nil or Prev~=emu:read32(CurrentPokemon) or PrevExp~=Game:getPokemonData(CurrentPokemon).experience or Frame < 5 or InBattle~=readBattleAddress() then
        -- console:log(string.format("8-bit: %i", readBattleAddress()))
        -- console:log(string.format("Number: %i", readBattleAddress()))
		printPokeStatus(Game, PrintBuffer, CurrentPokemon)
		Prev = emu:read32(CurrentPokemon)
		PrevExp = Game:getPokemonData(CurrentPokemon).experience
		Frame = Frame + 1
		InBattle = readBattleAddress()
		if Frame == 6 then 
            Frame = 0
        end
	end

    --PrintBuffer:moveCursor(0,1)
    
end

-- Pokemon Status
function printPokeStatus(game, buffer, pkm)
	buffer:clear()
	--[[
	local reverseCharmap = {}
	for k, v in pairs(Game._charmap) do
 	   reverseCharmap[v] = k
	end

	local testMove = "FIGHTING"
	for i = 1, #testMove do
    	local char = testMove:sub(i, i)
    	local byte = reverseCharmap[char]
    	print(char, string.format("0x%02X", byte))
	end
	--]]
	local currentPokemon = game:getPokemonData(pkm)
	local partyPokemon = {
        [0] = game:getPokemonData(Pokemon[1][2]),
        [1] = game:getPokemonData(Pokemon[2][2]),
        [2] = game:getPokemonData(Pokemon[3][2]),
        [3] = game:getPokemonData(Pokemon[4][2]),
        [4] = game:getPokemonData(Pokemon[5][2]),
        [5] = game:getPokemonData(Pokemon[6][2]),
    }

    for i = 0, 5 do
        local pokemon = partyPokemon[i]
        buffer:print(string.format("Party Pokemon %d: Species: %s\n",
        i + 1, pokemon.species))

    end

    --[[
	local inBattle = readBattleAddress()
	for i = 1, 4 do
		local move = currentPokemon.moves[i]
		buffer:print(string.format("Move Number %i: \n", move))
		--
		local moveName = game.moveNames + (currentPokemon.moves[i] * 13)
	    local name = game:toString(emu.memory.cart0:readRange(moveName, 12))

	    local moveEffectAddress = game.moveData + (currentPokemon.moves[i] * 12)
	    local effectID = emu.memory.cart0:read8(moveEffectAddress)

        --
		local damageAddress = game.moveData + (currentPokemon.moves[i] * 12) + 1
	    local damage = emu.memory.cart0:read8(damageAddress)

		local attackTypeID = game.moveData + (currentPokemon.moves[i] * 12) + 2
	    local attackTypeNumber = emu.memory.cart0:read8(attackTypeID)
		local attackTypeAddress = game.romTypesTable + (attackTypeNumber * 7)
		local attackType = game:toString(emu.memory.cart0:readRange(attackTypeAddress, 6))

		local accuracyAddress = game.moveData + (currentPokemon.moves[i] * 12) + 3
		local accuracy = emu.memory.cart0:read8(accuracyAddress)

		local target = game:getMoveTarget(currentPokemon.moves[i])

		buffer:print(string.format("Move %i: %-15s Damage: %-5s Type:%-7s Accuracy %-5s Effect ID: %-2s, Target: 0x%02X\n",
		i, name, damage, attackType, accuracy, effectID, target))
	end
    --]]
    local type1 = currentPokemon.type1
	local type2 = currentPokemon.type2
	buffer:print(string.format("Type 1: %-7s Type 2: %-7s\n", type1, type2))
	--[[
	local types = ""
	for i = 0, 17 do
		local ad = (0x0024F210) + (i * 7)
		types = types .. game:toString(emu.memory.cart0:readRange(ad, 6)) .. " "
		
	end	
	buffer:print(string.format("Type Names: %s\n", types))
	--]]

end


-- This is the start of the Socket functionality to communicate with our python code

local socket = socket:tcp()

function initializeSocketConnection()
    local ip_address = "127.0.0.1"
    local port = 65432
    socket:connect(ip_address, port)
    console:log("Connected our Socket to: " .. ip_address .. ":" .. port .. "\n")
    -- socket:send("Hi Holly and Ashley" .. "\r\n")
end

function EndSocketConnection()
    if socket then
        socket:close()
        console:log("Socket connection closed.")
    else
        console:log("No socket connection to close.")
    end
end

--[[
function send_test(message)
    if not socket then
        console:error("Socket is not initialized!")
        return
    end

    local success, err = socket:send(message .. "\r\n")
    if not success then
        console:error("Failed to send test message: " .. err)
    else
        console:log("Message " .. message .. " , sent successfully.")
    end
end
--]]
function SendMessageToServer(message)
    if not socket then
        console:error("Unable to send message because socket is invalid!")
        return
    end

    if message == nil then
        console:error("Given invalid message!")
        return
    end

    local success, err = socket:send(message .. '\r\n')
    if not success then
        console:error("Failed to send command: " .. err)
    else
        console:log("Sent command")
    end
end

function ReceiveFromSocket()
    if socket:hasdata() then
        local msg = socket:receive(128)
        if msg ~= nil then
            console:log("[+] server message: " .. msg)
            local parts = {}

            for part in string.gmatch(msg, "%S+") do
                table.insert(parts, part)
            end

            if #parts == 2 then
                local command = parts[1]
                local value = parts[2]

                if command == "PRESS_KEY" then
                    local input_num = tonumber(value)
                    console:log("PRESS KEY NUMBER: " .. input_num)
                    emu:addKey(input_num)

                    socket:send("KEY_PRESSED, " .. input_num .. "\r\n")

                elseif command == "RELEASE_KEY" then
                    local input_num = tonumber(value)
                    console:log("RELEASE KEY NUMBER: " .. input_num)
                    emu:clearKey(input_num)
                    socket:send("KEY_RELEASED" .. "\r\n")

                elseif command == "SELECT_MOVE" then
                    local moveIndex = tonumber(value)
                    console:log("Received move data from network" .. moveIndex)
                    local inPartyOrBag = emu:read8(0x200E728)
                    if inPartyOrBag == 0x38 then
                        console:log("Cursor is in the Pokemon Party, skipping move selection.")
                    elseif inPartyOrBag == 0x14 then
                        console:log("Cursor is in the Bag, skipping move selection.")
                    else  --not in party or bag 
                        CurrentMoveIndex = 1
                        IsAwaitingKeyAck = true
                        MovePath = Game:moveCursor(moveIndex)
                        --console:log("Path is: " .. table.concat(MovePath, ", "))

                        local key = MovePath[CurrentMoveIndex]
                        socket:send("PRESS_KEY," .. key .. "\r\n")     
                    end
                end
            else
                if msg == "SAVED_TURN_DATA" then
                    console:log("Turn data saved successfully!")
                    SocketCommunicating = false

                elseif msg == "ERROR" then
                    console:log("Socket communication failure!")
                    SocketCommunicating = false

                elseif msg == "KEY_PRESSED" then
                    console:log("IN KEY PRESSED")
                    if IsAwaitingKeyAck and CurrentMoveIndex < #MovePath then
                        CurrentMoveIndex = CurrentMoveIndex + 1
                        local key = MovePath[CurrentMoveIndex]

                        socket:send("PRESS_KEY," .. key .. "\r\n")
                    else   
                        console:log("All keys sent.")
                        IsAwaitingKeyAck = false
                        MovePath = {}
                        CurrentMoveIndex = 1
                    end
                end
            end
        end
    end

end


-- Initialize everything for the emulator

callbacks:add("keysRead", Input)
callbacks:add("frame", Update)
callbacks:add("start", InitializeGame)
callbacks:add("stop", EndSocketConnection)
callbacks:add("shutdown", EndSocketConnection)

socket:add("received", ReceiveFromSocket)

if emu then
	InitializeGame()
end