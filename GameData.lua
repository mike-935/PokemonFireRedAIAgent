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
--]]

-- Represents the ROM and the data on it
local GameData = {
    new = function (self, game)
        self.__index = self
        setmetatable(game, self) 
        return game
    end 
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

function InitializeGame()
    -- Represents the FireRed rom data
    Game = GameData:new({
        name = "FireRed (USA)",
        -- Address for where the first pokemon in the player's party is stored
        playerParty = 0x2024284,
        -- Address that stores the count of pokemon in the player's party (between 1 - 6)
        partyCount = 0x2024029,
        -- Address for the rom's table of the pokemon names
        romPokemonTable = 0x245EE0,
        -- Address for the enemy pokemon data 
        -- (could be a wild pokemon or the first pokemon of a enemy trainer)
        enemyParty = 0x0202402C,
        -- Address for my best guess at what signifies a battle
        inBattle = 0x020386B4,
        -- Address for where the rom stroes all the names of the moves
        moveNames = (0x00247110) - 13,
        -- Address for where the rom stores the move data
        moveData = (0x00250C80 - 12),
        -- Address for where the rom stores the pokemon types
        romTypesTable = (0x0024F210)
    })
    if not Game then
        console:error("Failed to initialize game data!")
        return
    else
        console:log("Game data initialized successfully!")
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
    LastPressedKey = nil
end

function ResetGame()
    PrintBuffer:clear()
    Frame = 0
    CurrentSelectedPokemon = 1
    LastPressedKey = nil
end


Right = 0x10
Start = 0x08
Left = 0x20

function Input()
    local selectedKey = emu:getKeys()
    local partyCount = emu:read8(Game.partyCount)
    if selectedKey ~= LastPressedKey then
        LastPressedKey = selectedKey
        if selectedKey == Right then
            CurrentSelectedPokemon = CurrentSelectedPokemon + 1
            if CurrentSelectedPokemon > partyCount then
                CurrentSelectedPokemon = 0
            end
        elseif selectedKey == Left then
            CurrentSelectedPokemon = CurrentSelectedPokemon - 1
            if CurrentSelectedPokemon < 1 then
                CurrentSelectedPokemon = partyCount
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

    if not emu:read32(Game.inBattle) then
        PrintBuffer:print("GameData battle address is invalid!")
        return
    end

    if Prev==nil or Prev~=emu:read32(CurrentPokemon) or PrevExp~=Game:getPokemonData(CurrentPokemon).experience or Frame < 5 or InBattleAddress~=emu:read32(Game.inBattle) then
		printPokeStatus(Game, PrintBuffer, CurrentPokemon)
		Prev = emu:read32(CurrentPokemon)
		PrevExp = Game:getPokemonData(CurrentPokemon).experience
		Frame = Frame + 1
		InBattleAddress = emu:read32(0x020386B4)
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
	local inBattle = emu:read32(game.inBattle)
	for i = 1, 4 do
		local move = currentPokemon.moves[i]
		-- buffer:print(string.format("Number %i: \n", move))	
		local moveName = game.moveNames + (currentPokemon.moves[i] * 13)
	    local name = game:toString(emu.memory.cart0:readRange(moveName, 12))
		local damageAddress = game.moveData + (currentPokemon.moves[i] * 12) + 1
	    local damage = emu.memory.cart0:read8(damageAddress)
		local attackTypeID = game.moveData + (currentPokemon.moves[i] * 12) + 2
	    local attackTypeNumber = emu.memory.cart0:read8(attackTypeID)
		local attackTypeAddress = game.romTypesTable + (attackTypeNumber * 7)
		local attackType = game:toString(emu.memory.cart0:readRange(attackTypeAddress, 6))
		buffer:print(string.format("Move %i: %-15s Damage: %-5s %-8s\n", i, name, damage, attackType))
	end	
	--[[
	local types = ""
	for i = 0, 17 do
		local ad = (0x0024F210) + (i * 7)
		types = types .. game:toString(emu.memory.cart0:readRange(ad, 6)) .. " "
		
	end	
	buffer:print(string.format("Type Names: %s\n", types))
	--]]

end

callbacks:add("keysRead", Input)
callbacks:add("frame", Update)
callbacks:add("reset", ResetGame)
callbacks:add("start", InitializeGame)

if emu then
	InitializeGame()
end