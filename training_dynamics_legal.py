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
from bagz import BagFileReader
from constants import CODERS
from scipy.stats import entropy
import jax.nn as jnn
from searchless_chess.src import utils
import matplotlib.pyplot as plt
from chess import svg as chess_svg


MODEL_NAME = 'BC_270M'
checkpoint_step =10000000
neural_engine = constants._build_neural_engine(MODEL_NAME, checkpoint_step)
stockfish_engine_top1 = constants.ENGINE_BUILDERS['stockfish']()

count_legal = 0
count_top1 = 0


INPUT_FILE =  "king_and_knights15.csv" 
_NUM_PUZZLES = 1000
puzzles = pd.read_csv(INPUT_FILE, nrows=_NUM_PUZZLES)


for puzzle_id, puzzle in tqdm(puzzles.iterrows(), total=_NUM_PUZZLES):
    fen = puzzle['FEN']
    #move = puzzle['Moves']
    board = chess.Board(fen)


    # Get transformer's best move
    #board.push(chess.Move.from_uci(puzzle["Moves"].split(' ')[0]))


    best_move_transformer = neural_engine.play(board)
    stockfish_move = stockfish_engine_top1.play(board).uci()
    in_top1 = (best_move_transformer.uci() == stockfish_move)
    if in_top1:
        if len(board.pieces(chess.KNIGHT, chess.WHITE)) == 15:
            print(board.fen())


    #Check if the move is legal
    legal = board.is_legal(best_move_transformer)
    if legal:
        count_legal += 1

print(count_legal)
#print(count_top1)

#stockfish_engine_top1.__del__()