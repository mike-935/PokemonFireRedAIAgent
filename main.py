import GameCommunicator as GC
import subprocess
import os
import platform

def find_rom():
    possible_roms = []
    for file in os.listdir("Emulator"):
        if file.endswith(".gba"):
            possible_roms.append(file)
    return possible_roms[0] if possible_roms else None

def find_save():
    possible_saves = []
    for file in os.listdir("Emulator"):
        split_filepath = file.split(".")
        if len(split_filepath) > 1 and split_filepath[-1].startswith("ss"):
            possible_saves.append(file)
    possible_saves.sort()
    print(possible_saves)
    return possible_saves[0] if possible_saves else None


def run_program():
    emulator_path = ""
    rom_path = os.path.join("Emulator", find_rom())
    script_path = "GameData.lua"
    save_path = os.path.join("Emulator", find_save())
    if platform.system() == "Linux":
        print("On Linux")
    elif platform.system() == "Darwin":
        emulator_path = "Emulator/mGBA.app/Contents/MacOS/mGBA"
        print("On MacOS")
    elif platform.system() == "Windows":
        emulator_path = "Emulator/mGBA.exe"
        print("On Windows")
    #print("Files exist?")
    #print("PokemonStatsGen3.lua:", os.path.isfile("./PokemonStatsGen3.lua"))
    #print("PokemonFireRed.gba:", os.path.isfile("Emulator/PokemonFireRed.gba"))

    if os.path.exists(emulator_path) and os.path.exists(rom_path):
        print("Emulator found at:", emulator_path)
        print("ROM found at:", rom_path)
    else:
        print("Failed to find emulator or ROM")

    process_args = [emulator_path, "--script",
         script_path, rom_path]

    if save_path and os.path.exists(save_path):
        print("User uses the .ss Save file and it was found found at:", save_path)
        process_args.append("--savestate")
        process_args.append(save_path)
    else:
        print("No save file found, using default save state.")


    process = subprocess.Popen(
        process_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        text=True
    )

if __name__ == "__main__":
    run_program()
    Communicator = GC.GameCommunicator()
    Communicator.run()