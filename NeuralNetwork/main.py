import sys
from GameCommunicator import Emu_Relay
import subprocess
import os
import platform

from glob import glob

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
Communicator = Emu_Relay.EmuRelay()

# Finds the first .gba file in the Emulator directory and returns its path.
def find_rom():
    possible_roms = []
    for rom_path in glob('Emulator/**/*.gba', recursive=True, root_dir=root_dir):
        possible_roms.append(rom_path)
    return os.path.join(root_dir, possible_roms[0]) if possible_roms else None

# Finds the first save state file in the Emulator directory and returns its path.
def find_save():
    possible_saves = []
    for savefile_path in glob('Emulator/**/*.ss[0-9]', recursive=True, root_dir=root_dir):
        possible_saves.append(savefile_path)
    possible_saves.sort()
    print(possible_saves)
    return os.path.join(root_dir, possible_saves[0]) if possible_saves else None

# Runs the mGBA emulator with the specified ROM and script.
# If a save state file is found, it will be used; otherwise, the default save state will be used.
# Supports Linux, macOS, and Windows platforms.
def run_program():
    emulator_path = ""
    rom_path = find_rom()
    script_path = "mGBAScripts/GameData.lua"
    save_path = find_save()
    if platform.system() == "Linux":
        emulator_path = "Emulator/mGBA.exe"
        print("On Linux")
    elif platform.system() == "Darwin":
        emulator_path = "Emulator/mGBA.app/Contents/MacOS/mGBA"
        print("On MacOS")
    elif platform.system() == "Windows":
        emulator_path = "Emulator/mGBA.exe"
        print("On Windows")

    emulator_path = os.path.join(root_dir, emulator_path)
    script_path = os.path.join(root_dir, script_path)

    if emulator_path is not None and os.path.exists(emulator_path):
        print(f"Emulator found at: {emulator_path}")
    else:
        print("Failed to find emulator or ROM")
        Communicator.close()
        sys.exit()

    if rom_path is not None and os.path.exists(rom_path):
        print(f"ROM found at: {rom_path}")
    else:
        print("Failed to find ROM")
        Communicator.close()
        sys.exit()


    process_args = [emulator_path, "--script",
         script_path, rom_path]

    if save_path:
        print(".ss Save file found")
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
    Communicator.run()
