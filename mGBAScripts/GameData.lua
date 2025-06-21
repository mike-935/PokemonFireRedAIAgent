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
    return pokemon
end


function readBattleAddress()
    -- This function reads the battle address to determine if a battle is ongoing
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

function GameData.sendTrainingData(game, currentPokemon)
    console:log("Sending to Python the turn to be saved...")

    local command = "SAVE_MOVE,"
    local battleData = {table.unpack(TurnData)}
    for i = 2, 6 do
        local partyPokemon = game:getPokemonData(Pokemon[i][2])
        local formattedPartyPokemon = game:formatPlayerPokemon(partyPokemon, false)
        for _, v in ipairs(formattedPartyPokemon) do
            table.insert(battleData, v)
        end
    end

    local playerDecision = game:getTurnDecision(currentPokemon)
    table.insert(battleData, playerDecision)

    local stringFormat = ""
    for _,data in ipairs(battleData) do
        stringFormat = stringFormat .. "%s,"
    end

    -- Remove the last comma
    stringFormat = string.sub(stringFormat, 1, -2)


    for i, v in ipairs(battleData) do
        console:log(string.format("Data[%d] = %s (type: %s)", i, tostring(v), type(v)))
    end


    local pokemonData = string.format(
        stringFormat,
        table.unpack(battleData)
    )

    console:log("Sending turn data: " .. command)
    -- SendMessageToServer(command)
    console:log("Finished sending turn data.")
end

function GameData.getTurnDecision(game, currentPokemon)
    -- first lets check for a move choice
    local decision = -1
    local chosenMoveIndex = -1
    for i = 1, 4 do
        local moveData = game:moveAsList(currentPokemon, i)
        if moveData[1] == getLastUsedMoveID() then
            chosenMoveIndex = i - 1
        end
    end

    console:log(string.format("Chosen move index is: %d", chosenMoveIndex))
    if chosenMoveIndex == -1 then
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
        -- if the pokemon switched, we return index + 3 to indicate a switch
        console:log(string.format("Switched to index %d which is treated as %d", leftOwn, leftOwn + 3))
        decision = leftOwn + 3
    else
        console:log("Chosen move index is: " .. chosenMoveIndex)
        decision = chosenMoveIndex
    end
    return decision
end

function GameData.saveTurnData(game, currentPokemon)
    local playerPokemonData = game:formatPlayerPokemon(currentPokemon, true)

    local opponentPokemon = game:getPokemonData(Game.enemyParty)
    local opponentPokemonData = game:formatOpponentPokemon(opponentPokemon)

    -- combine tables
    for _,data in ipairs(opponentPokemonData) do
        table.insert(playerPokemonData, data)
    end

    --[[
    for i = 2,6 do
        local pokemonAT = game:getPokemonData(Pokemon[i][2])
        if not pokemonAT then
            console:error(string.format("Pokemon %d data is nil!", i))
            return nil
        end
        if pokemonAT.species == 0 then
            console:log(string.format("Pokemon %d is empty, skipping...", i))
        end
        console:log(string.format("Pokemon %d: %s", i, game:getPokemonName(pokemonAT.species)))
    end
    --]]
    local stringFormat = ""
    for _,data in ipairs(playerPokemonData) do
        stringFormat = stringFormat .. "%s,"
    end

    -- Remove the last comma
    stringFormat = string.sub(stringFormat, 1, -2)

    --[[
    for i, v in ipairs(playerPokemonData) do
        console:log(string.format("Turn Data[%d] = %s (type: %s)", i, tostring(v), type(v)))
    end
    --]]
    return playerPokemonData
end

function GameData.requestAIMove(game, currentPokemon, trainingMode)
    console:log("Requesting AI move for current Pokemon...")

    local command = "REQUEST_AI_MOVE,"
    if trainingMode then
        command = "SAVE_MOVE,"
    end
    if not currentPokemon or currentPokemon == nil then
        console:error("Current Pokemon is nil when requesting ai move!")
        return
    end
    local pokemonData = game:formatPokemonData(currentPokemon, trainingMode)
    command = command .. pokemonData
    console:log("Sending AI move request with data: " .. command)
    SendMessageToServer(command)
    console:log("Finished sending AI move request.")
end

function GameData.formatPokemonData(game, playerPokemon, trainingMode)
    local playerPokemonData = game:formatPlayerPokemon(playerPokemon, true)

    -- structure of the data
    -- First is the Player Pokemon
    -- type, type2, level, status, currenthp, hp, atk, def, spatk, spdef, spd, moveXID, moveEffectID, moveXType, moveXDamage, moveXAccuracy, moveXpp
    -- Second is the Opponent Pokemon (appended to the end of above):
    -- type, type2, level, status, currenthp, hp, atk, def, spatk, spdef, spd
    -- 46 is last of opponent (spd)

    -- After the opponent pokemon we add each of the player's other pokemon in the format
    -- type, type2, level, status, currenthp, hp, atk, def, spatk, spdef, spd, switchable (58)

    -- And att the end we have the chosen move index or if you switched

    -- Get all the pokemon's moves in a table in the format:
    -- moveXID, moveEffectID, moveXType, moveXDamage, moveXAccuracy, moveXpp


    local opponentPokemon = game:getPokemonData(Game.enemyParty)
    local opponentPokemonData = game:formatOpponentPokemon(opponentPokemon)



    -- combine tables
    for _,data in ipairs(opponentPokemonData) do
        table.insert(playerPokemonData, data)
    end

    --[[
    for i = 2,6 do
        local pokemonAT = game:getPokemonData(Pokemon[i][2])
        if not pokemonAT then
            console:error(string.format("Pokemon %d data is nil!", i))
            return nil
        end
        if pokemonAT.species == 0 then
            console:log(string.format("Pokemon %d is empty, skipping...", i))
        end
        console:log(string.format("Pokemon %d: %s", i, game:getPokemonName(pokemonAT.species)))
    end
    --]]


    local chosenMoveIndex = -1
    for i = 1, 4 do
        local moveData = game:moveAsList(playerPokemon, i)
        if trainingMode then
            if moveData[1] == getLastUsedMoveID() then
                chosenMoveIndex = i - 1
            end
        end
    end

    if trainingMode then
        if chosenMoveIndex == -1 then
            console:log("Chosen move index is -1, this means the last used move was not found in the player's moves! or maybe it means switched")
        end
        console:log(string.format("Chosen move index is: %d", chosenMoveIndex))
        -- If training mode, append the chosen move index to the end of the player pokemon data
        table.insert(playerPokemonData, chosenMoveIndex)
    end

    local stringFormat = ""
    for _,data in ipairs(playerPokemonData) do
        stringFormat = stringFormat .. "%s,"
    end

    -- Remove the last comma
    stringFormat = string.sub(stringFormat, 1, -2)


    for i, v in ipairs(playerPokemonData) do
        console:log(string.format("Data[%d] = %s (type: %s)", i, tostring(v), type(v)))
    end


    local pokemonData = string.format(
        stringFormat,
        table.unpack(playerPokemonData)
    )

    return pokemonData
end

function GameData.formatPlayerPokemon(game, playerPokemon, active)
    -- Get all the player pokemon's data in the format of:
    -- type, type2, level, status, currnethp, hp, atk, def, spatk, spdef, spd, moveXID, moveEffectID, moveXType, moveXDamage, moveXAccuracy, moveXpp
    local playerPokemonData = {}
    if active then
        table.insert(playerPokemonData, playerPokemon.type1)
        table.insert(playerPokemonData, playerPokemon.type2)
        table.insert(playerPokemonData, playerPokemon.level)
        table.insert(playerPokemonData, playerPokemon.status)
        table.insert(playerPokemonData, playerPokemon.hp)
        local chosenMoveIndex = -1
        local moves = {}

        for i = 1, 4 do
            local moveData = game:moveAsList(playerPokemon, i)
            for _, v in ipairs(moveData) do
                table.insert(moves, v)
            end
        end

        local playerStartAddress = Game.playerBattleStruct + 24
        local playerStatStages = {
            -- hp
            emu:read8(playerStartAddress),
            -- attack
            emu:read8(playerStartAddress + 1),
            -- defense
            emu:read8(playerStartAddress + 2),
            -- Sp.Atk
            emu:read8(playerStartAddress + 4),
            -- Sp.Def
            emu:read8(playerStartAddress + 5),
            -- Spd
            emu:read8(playerStartAddress + 3),
            -- Accuracy
            -- Evasion
        }

        stringStats = {
            "HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Spd"
        }

        -- Add the stats of the player pokemon
        for i = 1,6 do
            table.insert(playerPokemonData, getEffectiveStat(playerPokemon.stats[i], playerStatStages[i] - 6))
        end

        -- Add the moves of the player pokemon
        for _,moveData in ipairs(moves) do
            table.insert(playerPokemonData, moveData)
        end

    else
        -- if not active we need to first check if the pokemon is alive or not
        -- if they are then we send type1, type2, level, status, currenthp, hp, atk, def, spatk, spdef, spd, 1
        -- if they are not then we send 0 for all stats
        if playerPokemon.species == 0 then
            console:log("Pokemon is null")
            for i = 1, 12 do
                table.insert(playerPokemonData, 0)
            end
        else
            table.insert(playerPokemonData, playerPokemon.type1)
            table.insert(playerPokemonData, playerPokemon.type2)
            table.insert(playerPokemonData, playerPokemon.level)
            table.insert(playerPokemonData, playerPokemon.status)
            table.insert(playerPokemonData, playerPokemon.hp)
            -- If not active, just add the base stats
            for i = 1,6 do
                table.insert(playerPokemonData, playerPokemon.stats[i])
            end
            table.insert(playerPokemonData, 1) -- 1 means the pokemon is alive and switchable
        end
    end

    return playerPokemonData
end

function GameData.formatOpponentPokemon(game, opponentPokemon)
    local opposingStatStageAddress = Game.opposingBattleStruct + 24
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
    -- type, type2, level, status, currenthp, hp, atk, def, spatk, spdef, spd
    local opponentPokemonData = {
        opponentPokemon.type1,
        opponentPokemon.type2,
        opponentPokemon.level,
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


function recreateParty()
    -- This function recreates the party from the GameData
    local party = {}
    local partyCount = emu:read8(Game.partyCount)
    for i = 1, partyCount do
        local pokemonData = Game:getPokemonData(Pokemon[i][2])
        if pokemonData.species ~= 0 then
            table.insert(i, pokemonData)
        end
    end
    return party
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
        playerBattleStruct = 0x2023BE4,
        -- Address representing the battle struct for the opposing pokemon
        opposingBattleStruct = 0x2023C3C,
        -- Address for the battle results
        battleResults = 0x3004F90,
        -- Address for the current battle's turn count
        turnCount = 0x3004FA3,
        -- Address for the last used move in the current battle
        lastUsedMove = 0x3004FB2,
        -- Address for the indexs of the 4 pokemon in a double battle
        battlerPartyIndexes = 0x2023BCE
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
    InBattleAddress = readBattleAddress()
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

    if InBattleAddress ~= readBattleAddress() and readBattleAddress() == 0 then
        console:log("First entering battle!")
        InBattleAddress = readBattleAddress()
        TurnData = Game:saveTurnData(currentActivePlayerPokemon)
    end
    -- maybe greater than 1, should just be != 0
    if CurrentTurn ~= getTurnCount() and readBattleAddress() == 0 and Game:getBattlersCount() == 2 then
        console:log(string.format("Turn Changed, Current Turn: %i, Previous Turn: %i, Last Used Move ID: %i", getTurnCount(), CurrentTurn, getLastUsedMoveID()))
        CurrentTurn = getTurnCount()
        if UseBattleAI then
            if not SocketCommunicating then
                console:log("Sending turn data...")
                SocketCommunicating = true
                Game:requestAIMove(currentActivePlayerPokemon, true)
                console:log("Turn Data sent!")
                -- On message receive turn off SocketCommunicating
            end
        end

        if TrainingMode and not SocketCommunicating then
            console:log("Sending turn data for training...")
            SocketCommunicating = true
            Game:sendTrainingData(currentActivePlayerPokemon)
            console:log("Turn Data sent for training!")
        end
        -- Save the turn data
        TurnData = Game:saveTurnData(currentActivePlayerPokemon)
        -- Save the current Active Pokemon
        LastActivePokemon = currentActivePlayerPokemon
    end

    --[[]
    if UseBattleAI then
        if not SocketCommunicating then
            console:log("AI will attempt to make a move!")
            SocketCommunicating = true
            playerPokemon = Game:getPokemonData(CurrentPokemon)
            Game:requestAIMove(playerPokemon, false)
            -- On message receive turn off SocketCommunicating
        end
    end
    --]]



    --[[
    if readBattleAddress() == 0 and UseBattleAI then
        if not SocketCommunicating then
            console:log("AI will attempt to make a move!")
            SocketCommunicating = true
            Game:requestAIMove(CurrentPokemon)
            -- On message receive turn off SocketCommunicating
        end
    end
    --]]

    -- need to check if we entered battle and if we are still in battle and when things/turn changes
    if Prev==nil or Prev~=emu:read32(CurrentPokemon) or PrevExp~=Game:getPokemonData(CurrentPokemon).experience or Frame < 5 or InBattleAddress~=readBattleAddress() then
        -- console:log(string.format("8-bit: %i", readBattleAddress()))
        -- console:log(string.format("Number: %i", readBattleAddress()))
		printPokeStatus(Game, PrintBuffer, CurrentPokemon)
		Prev = emu:read32(CurrentPokemon)
		PrevExp = Game:getPokemonData(CurrentPokemon).experience
		Frame = Frame + 1
		InBattleAddress = readBattleAddress()
		if Frame == 6 then 
            Frame = 0
        end
	end

    PrintBuffer:moveCursor(0,1)
    
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
    socket:send("Hi Holly and Ashley" .. "\r\n")
end

function EndSocketConnection()
    if socket then
        socket:close()
        console:log("Socket connection closed.")
    else
        console:log("No socket connection to close.")
    end
end

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

function SendMessageToServer(message)
    if not socket then
        console:error("Unable to send message because socket is invalid!")
        return
    end

    if message == nil then
        console:error("Given invalid message!")
        return
    end

    local success, err = socket:send(message .. "\r\n")
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
                    emu:addKey(input_num)
                    socket:send("received message" .. "\r\n")
                end

                if command == "RELEASE_KEY" then
                    local input_num = tonumber(value)
                    emu:clearKey(input_num)
                    socket:send("received message" .. "\r\n")
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