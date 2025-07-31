import sys
sys.path.append('searchless_chess/src')  # So imports work if running from project root

from bagz import BagFileReader, BagWriter
from constants import CODERS
import chess
import pandas as pd
import sys
import itertools
import subprocess
from tqdm import tqdm




def main():
    TRAIN_BAG_PATH = 'searchless_chess/data/train/behavioral_cloning_data.bag'
    same_color_path = 'same_color_puzzles.csv'
    more_pieces_path = 'more_pieces_puzzles.csv'

    reader = BagFileReader(TRAIN_BAG_PATH)

    num_of_rec = len(reader)
    piece_num = { chess.PAWN : 8, chess.ROOK : 2, chess.KNIGHT : 2, chess.BISHOP : 2, chess.QUEEN : 1}

    count_more_pieces = 0
    count_more_b = 0
    count_same_color = 0
    count_all_ood = 0
    count_written = 0
    
    for record in tqdm(reader):
            
        fen, move  = CODERS['behavioral_cloning'].decode(record)
        board = chess.Board(fen)

        bool_all_ood = False
        bool_more_pieces = False
        bool_same_color = False
        bool_more_b = False

        #Check if the number of piece type exceeds the limit
        for piece, num in piece_num.items():
            if (len(board.pieces(piece, chess.WHITE)) > num) or (len(board.pieces(piece, chess.BLACK)) > num):
                bool_more_pieces = True
                bool_all_ood = True
                if piece == chess.BISHOP:
                    bool_more_b = True

        #Check if there are 2 bishops on the same color
        if len(board.pieces(chess.BISHOP, chess.WHITE)) == 2:
            l = list(board.pieces(chess.BISHOP, chess.WHITE))
            if (chess.square_rank(l[0]) + chess.square_file(l[0]) + chess.square_rank(l[1]) + chess.square_file(l[1]))%2 == 0:
                bool_same_color = True
                bool_all_ood = True
        if len(board.pieces(chess.BISHOP, chess.BLACK)) == 2:
            l= list(board.pieces(chess.BISHOP, chess.BLACK))
            if (chess.square_rank(l[0]) + chess.square_file(l[0]) + chess.square_rank(l[1]) + chess.square_file(l[1]))%2 == 0:
                bool_same_color = True
                bool_all_ood = True


        if bool_more_pieces and not bool_more_b and not bool_same_color:
            with open(more_pieces_path, "a") as outfile:
                outfile.write(fen + ',' + move + '\n')
                count_more_pieces += 1
        if not bool_more_pieces and bool_same_color:
            with open(same_color_path, "a") as outfile:
                outfile.write(fen + ',' + move + '\n')
                count_same_color += 1

    print(count_more_pieces)
    print(count_same_color)

if __name__ == '__main__':
    
    main()