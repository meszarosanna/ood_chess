#Copyright 2025 Anna Meszaros

from shutil import move
import chess
import numpy as np
from searchless_chess.src.engines import constants
from searchless_chess.src import utils
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import PowerNorm 
import cairosvg


MODEL_NAME = 'BC_270M'
checkpoint_step = 10000000
neural_engine = constants._build_neural_engine(MODEL_NAME, checkpoint_step)



board = chess.Board("r1b2rk1/1p3ppp/p1n1qb2/8/3N1B2/P5P1/1PQ1PPBP/3R1RK1 b - - 10 18")


total_log_probs = neural_engine.analyse(board)
total_probs = np.exp(np.array(total_log_probs))
all_moves = [chess.Move.from_uci(utils.ACTION_TO_MOVE[i]) for i in range(len(utils.ACTION_TO_MOVE))]

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

    
# Map heatmap values to colors
norm = PowerNorm(gamma = 0.7, vmin=0, vmax=heatmap.max()) #heatmap.min()
cmap = plt.cm.viridis
rgba = cmap(norm(heatmap))
hex_colors = [[matplotlib.colors.rgb2hex(rgba[i, j]) for j in range(8)] for i in range(8)]
hex_colors_flipped = hex_colors[::-1]

# Flatten row by row (left to right, bottom to top)
flat_colors = [color for row in hex_colors_flipped for color in row]

# Dictionary: bottom-left (a1) = 0, top-right (h8) = 63
chess_square_colors = {i: flat_colors[i] for i in range(64)}

#Save as PDF
svg = chess.svg.board(
    board,
    fill=chess_square_colors,
    size=400
)
cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to="training_board.pdf")


    




    