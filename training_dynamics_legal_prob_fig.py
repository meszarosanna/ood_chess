from shutil import move
import sys
sys.path.append('searchless_chess/src')  # So imports work if running from project root

import chess
import chess.pgn
import pandas as pd
import io
from tqdm import tqdm
import itertools
import time
import math
import numpy as np
from best_move_transformer import get_best_move as get_transformer_move
from best_move_transformer import load_transformer_model
from best_move_stockfish import get_best_move as get_stockfish_moves
from searchless_chess.src.engines import constants
from searchless_chess.src.engines import engine
from bagz import BagFileReader
from constants import CODERS
from scipy.stats import entropy
import jax.nn as jnn
from searchless_chess.src import utils
import matplotlib.pyplot as plt
from chess import svg as chess_svg
import csv


MODEL_NAME = 'BC_270M'
_NUM_PUZZLES = 1000
#list_of_steps = [0, 800, 1000, 2000, 3000, 100000, 400000, 800000, 1200000]
#list_of_steps = [100, 200, 300, 400, 500, 600, 700, 900]
#list_of_steps = [4000, 5000, 6000, 7000, 8000, 9000, 10000, 200000]
list_of_steps = [300000]
def eval_puzzles(filename, steps):
    neural_engine = constants._build_neural_engine(MODEL_NAME, steps)
    puzzles = pd.read_csv(filename, nrows=_NUM_PUZZLES)

    legal_probs_list = []

    for puzzle_id, puzzle in tqdm(puzzles.iterrows(), total=_NUM_PUZZLES):
        fen = puzzle['FEN']
        board = chess.Board(fen)
        board.push(chess.Move.from_uci(puzzle["Moves"].split(' ')[0]))

        total_log_probs = neural_engine.analyse(board)
        total_probs = np.exp(np.array(total_log_probs))

        sorted_legal_moves = engine.get_ordered_legal_moves(board)
        legal_actions = [utils.MOVE_TO_ACTION[x.uci()] for x in sorted_legal_moves]
        legal_actions = np.array(legal_actions, dtype=np.int32)
        legal_probs = total_probs[legal_actions]

        legal_probs_list.append(sum(legal_probs))
        
    return sum(legal_probs_list)/len(legal_probs_list)

id_legal = []
ood_legal = []   

for step in list_of_steps:
    print(f"Evaluating step {step}")
    id_legal.append(eval_puzzles("id_puzzles.csv", step))
    ood_legal.append(eval_puzzles("ood_puzzles.csv", step))

print(id_legal)
print(ood_legal)

results = [id_legal, ood_legal]

with open("legal_prob_results.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)
