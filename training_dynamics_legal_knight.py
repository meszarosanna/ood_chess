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
checkpoint_step = 1000
neural_engine = constants._build_neural_engine(MODEL_NAME, checkpoint_step)


INPUT_FILE =  "id_puzzles.csv" #'searchless_chess/data/test/filtered_behavioral_cloning_data.bag'
_NUM_PUZZLES = 1
puzzles = pd.read_csv(INPUT_FILE, nrows=_NUM_PUZZLES)

#reader = BagFileReader(INPUT_FILE)
#num_of_rec = len(reader)

all_moves = [chess.Move.from_uci(utils.ACTION_TO_MOVE[i]) for i in range(len(utils.ACTION_TO_MOVE))]

def knight(move):
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    
    df = abs(to_file - from_file)
    dr = abs(to_rank - from_rank)

    return (df == 2 and dr == 1) or (df == 1 and dr == 2)

def pawn(move, color):
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    
    # Pawns move differently based on color
    forward = 1 if color == chess.WHITE else -1
    start_rank = 1 if color == chess.WHITE else 6
    
    df = to_file - from_file
    dr = to_rank - from_rank
    
    if df == 0 and dr == forward:
        return True
    elif df == 0 and dr == 2 * forward and from_rank == start_rank:
        return True
    elif abs(df) == 1 and dr == forward:
        return True
    else:
        return False

def rook(move):
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    
    # Rook moves along rank or file
    return from_file == to_file or from_rank == to_rank

def bishop(move):
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    
    # Bishop moves diagonally: |df| == |dr|
    df = abs(to_file - from_file)
    dr = abs(to_rank - from_rank)
    
    return df == dr 

def queen(move):
    return bishop(move) or rook(move)

def king(move):
    return NotImplementedError


for puzzle_id, puzzle in tqdm(puzzles.iterrows(), total=_NUM_PUZZLES):

    board = chess.Board("4k3/8/8/8/8/8/8/R3K2R")
    total_log_probs = neural_engine.analyse(board)
    total_probs = np.exp(np.array(total_log_probs))

    king_moves = [move for move in board.legal_moves if board.piece_type_at(move.from_square) == chess.KING]
    
    legal_moves = [move for move in board.legal_moves if board.piece_type_at(move.from_square) == chess.ROOK]
    

    all_piece = [move for move in all_moves if rook(move) and move not in king_moves]

    illegal_moves = [move for move in all_piece if move not in legal_moves]

    legal_moves = [utils.MOVE_TO_ACTION[x.uci()] for x in legal_moves]
    illegal_moves = [utils.MOVE_TO_ACTION[x.uci()] for x in illegal_moves]

    legal_moves = np.array(legal_moves, dtype=np.int32)
    illegal_moves = np.array(illegal_moves, dtype=np.int32)

    legal_probs = total_probs[legal_moves]
    illegal_probs = total_probs[illegal_moves]
    

    print(sum(legal_probs)/(sum(legal_probs)+sum(illegal_probs)))

    




    