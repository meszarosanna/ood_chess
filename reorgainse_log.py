import pandas as pd

with open("log_tournament_chess9603.csv", "r") as f:
    for line in f:
        line = line.strip()
        if "Starting" in line or "End" in line or "Playing" in line or "," in line:
            with open("log_tournament_chess9603_new.csv", 'a') as file:
                file.write(f'{line}\n')
        else:
            with open("log_tournament_chess9603_new.csv", 'a') as file:
                file.write(f'{line},')