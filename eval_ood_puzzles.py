import chess
import chess.pgn
import pandas as pd
import io
from best_move_transformer import get_best_move as get_transformer_move
from best_move_transformer import load_transformer_model
from best_move_stockfish import get_best_move as get_stockfish_moves

MODEL_NAME = '270M' 
neural_engine = load_transformer_model(MODEL_NAME)

# Stockfish parameters
STOCKFISH_DEPTH = None
STOCKFISH_TIME = 0.5
STOCKFISH_MULTIPV = 5

count_legal = 0
count_top5 = 0
count_top3 = 0
count_top1 = 0
count_top10 = 0
count_entire = 0
count_stockfish_error_3 = 0  # Counter for IndexError
count_stockfish_error_5 = 0  # Counter for IndexError
count_stockfish_error_10 = 0  # Counter for IndexError

INPUT_FILE = 'searchless_chess/data/puzzles.csv'
_NUM_PUZZLES = 1000
puzzles = pd.read_csv(INPUT_FILE, nrows=_NUM_PUZZLES)

def eval_entire_sequence(board):
    for move_idx, move in enumerate(puzzle["Moves"].split(' ')[1:]):
        if move_idx % 2 == 0:
            predicted_move = neural_engine.play(board).uci()
            # Lichess puzzles consider all mate-in-1 moves as correct, so we need to
            # check if the `predicted_move` results in a checkmate if it differs from
            # the solution.
            if move != predicted_move:
                if board.is_legal(chess.Move.from_uci(predicted_move)):
                    board.push(chess.Move.from_uci(predicted_move))
                    return board.is_checkmate()
                else:
                    return False
        board.push(chess.Move.from_uci(move))
    return True
 

for puzzle_id, puzzle in puzzles.iterrows():
    fen = puzzle["FEN"]
    # Get transformer's best move
    board = chess.Board(fen)
    board.push(chess.Move.from_uci(puzzle["Moves"].split(' ')[0]))
    fen = board.fen()

    best_move_transformer = neural_engine.play(board)

    #Check if the move is legal
    if board.is_legal(best_move_transformer):
        count_legal += 1


    # Check if transformer's move is in Stockfish's top 1,3,5,10
        
    stockfish_moves_1 = get_stockfish_moves(fen, STOCKFISH_DEPTH, STOCKFISH_TIME, 1, win_probab=False)
    in_top1 = best_move_transformer.uci() in stockfish_moves_1
    if in_top1:
        count_top1 += 1

    try:
        stockfish_moves_3 = get_stockfish_moves(fen, STOCKFISH_DEPTH, STOCKFISH_TIME, 3, win_probab=False)
        in_top3 = best_move_transformer.uci() in stockfish_moves_3
        if in_top3:
            count_top3 += 1
    except IndexError:
        count_stockfish_error_3 += 1
    try:
        stockfish_moves_5 = get_stockfish_moves(fen, STOCKFISH_DEPTH, STOCKFISH_TIME, 5, win_probab=False)
        in_top5 = best_move_transformer.uci() in stockfish_moves_5
        if in_top5:
            count_top5 += 1
    except IndexError:
        count_stockfish_error_5 += 1
    try:
        stockfish_moves_10 = get_stockfish_moves(fen, STOCKFISH_DEPTH, STOCKFISH_TIME, 10, win_probab=False)
        in_top10 = best_move_transformer.uci() in stockfish_moves_10
        if in_top10:
            count_top10 += 1
    except IndexError:
        count_stockfish_error_10 += 1
    
    
    
    #Check if matches the entire solution
    if eval_entire_sequence(board):
        count_entire += 1 

    if puzzle_id == 9:
        print("When predicting the next move, from 10 puzzles:")
        print(f'{count_legal} are legal')
        print(f'{count_top1} are in top1 of Stockfish')
        print(f'{count_top3} are in top3 of Stockfish')
        print(f'{count_top5} are in top5 of Stockfish')
        print(f'{count_top10} are in top10 of Stockfish')
        print(f'{count_entire} are matching the entire sequence of the solution')
        #print(count_legal)
        #print(count_top1)   
        #print(count_top3)
        #print(count_top5)
        #print(count_top10)
        #print(count_entire)
        print("And Stockfish coundn't give")
        print(f' top3 moves in {count_stockfish_error_3} cases')
        print(f' top5 moves in {count_stockfish_error_5} cases')
        print(f' top10 moves in {count_stockfish_error_10} cases')
        #print(count_stockfish_error_3)
        #print(count_stockfish_error_5)  
        #print(count_stockfish_error_10)
        break

