import pandas as pd
import chess
import chess.pgn
import io

# Path to the puzzles CSV file
PUZZLES_PATH = 'ood_puzzles.csv'
NUM_PUZZLES = 12379


def main():
    # Load only the header to get column names
    puzzles = pd.read_csv(PUZZLES_PATH, nrows=NUM_PUZZLES)
    piece_num = { chess.PAWN : 8, chess.ROOK : 2, chess.KNIGHT : 2, chess.BISHOP : 2, chess.QUEEN : 1}
    
    
    count_all_ood = 0
    count_more_pieces = 0
    count_more_b = 0
    count_same_color = 0


    for puzzle_id, puzzle in puzzles.iterrows():
        bool_all_ood = False
        bool_more_pieces = False
        bool_same_color = False

        board = chess.Board(puzzle["FEN"])
        board.push(chess.Move.from_uci(puzzle["Moves"].split(' ')[0]))


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
        
    print(count_all_ood)
    print(count_more_pieces)
    print(count_more_b)
    print(count_same_color)

if __name__ == '__main__':
    main()