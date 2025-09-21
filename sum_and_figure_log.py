import pandas as pd
import matplotlib.pyplot as plt
import math

illegal = 0
all_moves =0

with open("log_tournament_chess9603.csv", "r") as f:
    for i, line in enumerate(f):
        if i == 20:
            break
        line = line.strip()
        if "Starting" in line or "End" in line or "Playing" in line:
            pass
        else:
            split = line.split(',')
            illegal += int(split[-2])
            all_moves += int(split[-1])
            split = split[:-2]
            scores = []
            for e in split:
                if '#+' in e:
                    scores.append(1)
                    scores.append(0)
                elif '#-' in e:
                    scores.append(0)
                    scores.append(1)
                else:
                    scores.append(1/(1+math.exp(-0.00368208*int(e))))
            #splitted = [0 if "#" in e else 1-1/(1+math.exp(-0.00368208*int(e))) for e in splitted]
            #splitted.append(1)
            if i%2 == 0:
                scores = scores[1::2]
            else:
                scores = scores[::2]
            
            plt.plot(scores)
            plt.title("Chess960 with openings: BC_270M vs Stockfish1")
            plt.savefig("plot-BCvsS1chess960openings.png")
        

#38 103846 for classic -> 99.96 legal
#270 113202 for chess960 -> 99.76 legal


            