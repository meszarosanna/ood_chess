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
import matplotlib
from chess import svg as chess_svg
from matplotlib.colors import PowerNorm 


MODEL_NAME = 'BC_270M'
checkpoint_step = 10000000
neural_engine = constants._build_neural_engine(MODEL_NAME, checkpoint_step)




INPUT_FILE =  "more_pieces_puzzles.csv" #'searchless_chess/data/test/filtered_behavioral_cloning_data.bag'
_NUM_PUZZLES = 20
puzzles = pd.read_csv(INPUT_FILE, nrows=_NUM_PUZZLES)

#reader = BagFileReader(INPUT_FILE)
#num_of_rec = len(reader) 

for puzzle_id, puzzle in tqdm(puzzles.iterrows(), total=_NUM_PUZZLES):
#for record in tqdm(itertools.islice(reader, _NUM_PUZZLES), total=_NUM_PUZZLES):
    #fen, move  = CODERS['behavioral_cloning'].decode(record)
    fen = puzzle['FEN']
    move = puzzle['Moves'] #.split(' ')[1]
    board = chess.Board(fen)


    # Get transformer's best move
    #board.push(chess.Move.from_uci(puzzle["Moves"].split(' ')[0]))

    #board = chess.Board("2k5/1pp1r2Q/pr2p3/3p1p2/q2b4/P2NP3/1PP3PP/K3RR2 w - - 0 26")
    
    total_log_probs = neural_engine.analyse(board)
    total_probs = np.exp(np.array(total_log_probs))
    all_moves = [chess.Move.from_uci(utils.ACTION_TO_MOVE[i]) for i in range(len(utils.ACTION_TO_MOVE))]

    #best_move = neural_engine.play(board)
    
    
    # Print top-5 most probable moves (by action probability)
    #top_k = 5
    #top_indices = np.argsort(total_probs)[-top_k:][::-1]
    #print('Top 5 moves:')
    #for rank_idx, move_idx in enumerate(top_indices, start=1):
    #    pr = float(total_probs[move_idx])
    #    mv = all_moves[move_idx]
    #    print(f"{rank_idx}. {mv.uci()}  {pr:.6f}")




    # Build from-square heatmap from move probabilities
    probs_by_from_square = np.zeros(64, dtype=float)
    for move, prob in zip(all_moves, total_probs):
        from_sq = move.from_square
        probs_by_from_square[from_sq] += float(prob)

    # Create 8x8 heatmap array from from-square probabilities
    heatmap = np.zeros((8, 8), dtype=float)
    for sq in range(64):
        row = 7 - chess.square_rank(sq)
        col = chess.square_file(sq)
        heatmap[row, col] = probs_by_from_square[sq]

    norm = PowerNorm(gamma = 0.8, vmin=heatmap.min(), vmax=heatmap.max())

    cmap = plt.cm.viridis
    rgba = cmap(norm(heatmap))
    hex_colors = [[matplotlib.colors.rgb2hex(rgba[i, j]) for j in range(8)] for i in range(8)]

    hex_colors_flipped = hex_colors[::-1]

    # Flatten row by row (left to right, bottom to top)
    flat_colors = [color for row in hex_colors_flipped for color in row]

    # Dictionary: bottom-left (a1) = 0, top-right (h8) = 63
    chess_square_colors = {i: flat_colors[i] for i in range(64)}


    svg = chess.svg.board(
        board,
        fill=chess_square_colors,
        #arrows=[chess.svg.Arrow(best_move.from_square, best_move.to_square, color="#02189380")],
        size=400
    )

    
    # 5. Save SVG
    with open("heatmap.svg", "w") as f:
        f.write(svg)



    plt.figure(figsize=(6, 6))
    im = plt.imshow(heatmap, cmap='viridis', origin='upper')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Probability mass')
    plt.xticks(ticks=range(8), labels=list('abcdefgh'))
    plt.yticks(ticks=range(8), labels=list('87654321'))
    plt.title('From-square probability heatmap')
    plt.xlabel('File')
    plt.ylabel('Rank')
    plt.tight_layout()
    plt.savefig('heatmap.png')

    time.sleep(5)

    




    