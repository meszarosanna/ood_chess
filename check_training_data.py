import sys
sys.path.append('searchless_chess/src')  # So imports work if running from project root

from bagz import BagFileReader
from constants import CODERS
import chess
import pandas as pd
import sys
import itertools
import subprocess

PUZZLES_PATH = 'ood_puzzles.csv'
NUM_PUZZLES = 12379


FROM = 525200000


def main():
    TRAIN_BAG_PATH = 'searchless_chess/data/train/behavioral_cloning_data.bag'
    puzzles = pd.read_csv(PUZZLES_PATH, nrows=NUM_PUZZLES)

    reader = BagFileReader(TRAIN_BAG_PATH)
    num_of_rec = len(reader)
    piece_num = { chess.PAWN : 8, chess.ROOK : 2, chess.KNIGHT : 2, chess.BISHOP : 2, chess.QUEEN : 1}

    count_more_pieces = 0
    count_more_b = 0
    count_same_color = 0
    count_all_ood = 0
    count_puzzle = 0
    
    for i, record in enumerate(itertools.islice(reader, FROM, None)):
        if i%10000 == 0:
            print(i)
            
        fen, move  = CODERS['behavioral_cloning'].decode(record)
        board = chess.Board(fen)
        data_fen = board.board_fen()
        bool_all_ood = False
        bool_more_pieces = False
        bool_same_color = False

        #Check if the number of piece type exceeds the limit
        for piece, num in piece_num.items():
            if (len(board.pieces(piece, chess.WHITE)) > num) or (len(board.pieces(piece, chess.BLACK)) > num):
                bool_more_pieces = True
                bool_all_ood = True
                if piece == chess.BISHOP:
                    count_more_b += 1
        if bool_more_pieces:
            count_more_pieces += 1

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
        if bool_same_color:
            count_same_color +=1


        if bool_all_ood:
            count_all_ood += 1
            
            for puzzle_id, puzzle in puzzles.iterrows():
                board = chess.Board(puzzle["FEN"])
                board.push(chess.Move.from_uci(puzzle["Moves"].split(' ')[0]))
                puzzle_fen = board.board_fen()
                if puzzle_fen == data_fen:
                    count_puzzle += 1
                    with open("ood.csv" , "a") as outfile:
                        outfile.write(puzzle_fen + '\n')
            
    filename = f"details_{FROM}.csv"
    with open(filename, "a") as outfile:
        outfile.write(str(count_all_ood) + '\n' + str(count_more_pieces) + '\n' + str(count_more_b) + '\n' + str(count_same_color) + '\n')
        
    #print(num_of_rec)
    #print(count_all_ood)
    #print(count_more_pieces)
    #print(count_more_b)
    #print(count_same_color)
    #print(count_puzzle)

if __name__ == '__main__':
    
        # Worker mode: FROM argument is present, just run main()
    main()