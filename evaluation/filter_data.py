#Copyright 2025 Anna Meszaros

from searchless_chess.src.bagz import BagFileReader, BagWriter
from searchless_chess.src.constants import CODERS
import chess
from tqdm import tqdm


def main():
    TRAIN_BAG_PATH = 'searchless_chess/data/train/behavioral_cloning_data.bag' #for test data, change 'train' to 'test'
    FILTERED_BAG_PATH = 'searchless_chess/data/train/filtered_behavioral_cloning_data.bag' #for test data, change 'train' to 'test'

    reader = BagFileReader(TRAIN_BAG_PATH)
    writer = BagWriter(FILTERED_BAG_PATH)

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
        else:
            # Write the current (fen, move) to the filtered bag file
            writer.write(CODERS['behavioral_cloning'].encode((fen, move)))
            count_written += 1

    # Close the writer 
    writer.close()     

    # Write details to a file
    filename = f"details.csv"
    with open(filename, "a") as outfile:
        outfile.write("Number of all records: " + str(num_of_rec) + '\n' + "Number of records written in the file: " + str(count_written) + '\n' + "Number of OOD boards (it should be equal to the previous number): " + str(count_all_ood) + '\n' + "Number of boards with more pieces: " + str(count_more_pieces) + '\n' + "Number of boards with more bishops: " + str(count_more_b) + '\n' + "Number of boards with 2 bishops on the same color: " + str(count_same_color) + '\n')
        

if __name__ == '__main__':
    
    main()