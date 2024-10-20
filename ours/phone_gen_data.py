import os
import subprocess

def create_folders(base_directory):
    characters = "0123456789abcdefghijklmnopqrstuvwxyz"
    for char in characters:
        folder_path = os.path.join(base_directory, char)
        os.makedirs(folder_path, exist_ok=True)

def execute_commands():
    characters = "0123456789abcdefghijklmnopqrstuvwxyz"
    for char in characters:
        print("-------------------------", char)
        command = f"python .\\phone.py ..\\dataset\\MBPWavs\\{char}.wav ..\\new_dataset_phone\\{char}\\"
        subprocess.run(command, shell=True)

create_folders("../new_dataset_phone")
execute_commands()