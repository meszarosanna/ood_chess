#Copyright 2025 Anna Meszaros

"""Creates the more_pieces_puzzles and same_color_puzzles datasets."""

from searchless_chess.src.bagz import BagFileReader
from searchless_chess.src.constants import CODERS
import chess
from tqdm import tqdm

def main():
    TRAIN_BAG_PATH = 'searchless_chess/data/train/behavioral_cloning_data.bag'
    same_color_path = 'same_color_puzzles.csv'
    more_pieces_path = 'more_pieces_puzzles.csv'

    reader = BagFileReader(TRAIN_BAG_PATH)

    piece_num = { chess.PAWN : 8, chess.ROOK : 2, chess.KNIGHT : 2, chess.BISHOP : 2, chess.QUEEN : 1}

    count_more_pieces = 0
    count_same_color = 0
  
    
    for record in tqdm(reader):
            
        fen, move  = CODERS['behavioral_cloning'].decode(record)
        board = chess.Board(fen)

        bool_more_pieces = False
        bool_same_color = False
        bool_more_b = False

        #Check if the number of piece type exceeds the limit
        for piece, num in piece_num.items():
            if (len(board.pieces(piece, chess.WHITE)) > num) or (len(board.pieces(piece, chess.BLACK)) > num):
                bool_more_pieces = True
                if piece == chess.BISHOP:
                    bool_more_b = True

        #Check if there are 2 bishops on the same color
        if len(board.pieces(chess.BISHOP, chess.WHITE)) == 2:
            l = list(board.pieces(chess.BISHOP, chess.WHITE))
            if (chess.square_rank(l[0]) + chess.square_file(l[0]) + chess.square_rank(l[1]) + chess.square_file(l[1]))%2 == 0:
                bool_same_color = True
        if len(board.pieces(chess.BISHOP, chess.BLACK)) == 2:
            l= list(board.pieces(chess.BISHOP, chess.BLACK))
            if (chess.square_rank(l[0]) + chess.square_file(l[0]) + chess.square_rank(l[1]) + chess.square_file(l[1]))%2 == 0:
                bool_same_color = True



        if bool_more_pieces and not bool_more_b and not bool_same_color:
            with open(more_pieces_path, "a") as outfile:
                outfile.write(fen + ',' + move + '\n')
                count_more_pieces += 1
        if not bool_more_pieces and bool_same_color:
            with open(same_color_path, "a") as outfile:
                outfile.write(fen + ',' + move + '\n')
                count_same_color += 1

    print(f"{count_more_pieces} positions with more pieces added to {more_pieces_path}")
    print(f"{count_same_color} positions with same color bishops added to {same_color_path}")

if __name__ == '__main__':
    
    main()