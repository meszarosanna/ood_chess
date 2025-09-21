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
import csv

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
all_moves = [chess.Move.from_uci(utils.ACTION_TO_MOVE[i]) for i in range(len(utils.ACTION_TO_MOVE))]
#list_of_steps = [0, 800, 1000, 2000, 3000, 100000, 400000, 800000, 1200000]
#list_of_steps = [100, 200, 300, 400, 500, 600, 700, 900]
#list_of_steps  = [4000, 5000, 6000, 7000, 8000, 9000, 10000, 200000]
list_of_steps = [300000]
def knight(move):
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    
    df = abs(to_file - from_file)
    dr = abs(to_rank - from_rank)

    return (df == 2 and dr == 1) or (df == 1 and dr == 2)

def pawn(move, color=chess.WHITE):
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
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)

    df = abs(to_file - from_file)
    dr = abs(to_rank - from_rank)

    return df <= 1 and dr <= 1 #doesn't consider castling

def eval_board(steps, fen, piece_type, piece_fun, not_king):
    
    neural_engine = constants._build_neural_engine(MODEL_NAME, steps)

    board = chess.Board(fen)
   
    total_log_probs = neural_engine.analyse(board)
    total_probs = np.exp(np.array(total_log_probs))

    king_moves = [move for move in board.legal_moves if board.piece_type_at(move.from_square) == chess.KING]
    
    #total_probs[king_moves_prob] = 0
    #total_probs = np.array(total_probs)
    #total_probs = total_probs / sum(total_probs)


    legal_moves = [move for move in board.legal_moves if board.piece_type_at(move.from_square) == piece_type]
    if not_king:
        all_piece = [move for move in all_moves if piece_fun(move) and move not in king_moves]
    else:
        all_piece = [move for move in all_moves if piece_fun(move)]
    illegal_moves = [move for move in all_piece if move not in legal_moves]

    legal_moves = [utils.MOVE_TO_ACTION[x.uci()] for x in legal_moves]
    illegal_moves = [utils.MOVE_TO_ACTION[x.uci()] for x in illegal_moves]

    legal_moves = np.array(legal_moves, dtype=np.int32)
    illegal_moves = np.array(illegal_moves, dtype=np.int32)

    legal_probs = total_probs[legal_moves]
    illegal_probs = total_probs[illegal_moves]
    return sum(legal_probs)/(sum(legal_probs)+sum(illegal_probs))


knightk_fen = "4k3/8/8/8/8/8/8/1N2K1N1"
rookk_fen = "4k3/8/8/8/8/8/8/R3K2R"
bishopk_fen = "4k3/8/8/8/8/8/8/2B1KB2"
queenk_fen = "4k3/8/8/8/8/8/8/3QK3"
pawnk_fen = "4k3/8/8/8/8/8/PPPPPPPP/4K3"
king_fen = "4k3/8/8/8/8/8/8/4K3"

knight_fen = "4k3/8/8/8/8/8/8/1N4N1"
rook_fen = "4k3/8/8/8/8/8/8/R6R"
bishop_fen = "4k3/8/8/8/8/8/8/2B2B2"
queen_fen = "4k3/8/8/8/8/8/8/3Q4"
pawn_fen = "4k3/8/8/8/8/8/PPPPPPPP/8"

result_nkn = []
result_rkr = []
result_bkb = []
result_qk = []
result_pkp = []
result_k = []

result_nn =[]
result_rr = []
result_bb = []
result_q = []
result_pp = []

for step in list_of_steps:
    print(f"Evaluating step {step}")
    result_nkn.append(eval_board(step, knightk_fen, chess.KNIGHT, knight, False))
    result_nn.append(eval_board(step, knight_fen, chess.KNIGHT, knight, False))

    result_rkr.append(eval_board(step, rookk_fen, chess.ROOK, rook, True))
    result_rr.append(eval_board(step, rook_fen, chess.ROOK, rook, True))

    result_bkb.append(eval_board(step, bishopk_fen, chess.BISHOP, bishop, True))
    result_bb.append(eval_board(step, bishop_fen, chess.BISHOP, bishop, True))

    result_qk.append(eval_board(step, queenk_fen, chess.QUEEN, queen, True))
    result_q.append(eval_board(step, queen_fen, chess.QUEEN, queen, True))

    result_pkp.append(eval_board(step, pawnk_fen, chess.PAWN, pawn, True))
    result_pp.append(eval_board(step, pawn_fen, chess.PAWN, pawn, True))

    result_k.append(eval_board(step, king_fen, chess.KING, king, False))

print(result_nkn)
print(result_nn)
print(result_rkr)
print(result_rr)
print(result_bkb)
print(result_bb)
print(result_qk)
print(result_q)
print(result_pkp)
print(result_pp)
print(result_k)

results = [result_nkn, result_nn, result_rkr, result_rr, result_bkb, result_bb, result_qk, result_q, result_pkp, result_pp, result_k]

with open("piece_results.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)
    


    
