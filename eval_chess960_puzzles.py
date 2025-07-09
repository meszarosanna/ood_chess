import chess
from best_move_transformer import get_best_move as get_transformer_move
from best_move_transformer import load_transformer_model
from best_move_stockfish import get_best_move as get_stockfish_moves

INPUT_FILE = 'ood_puzzles_all_ordering.csv'
MODEL_NAME = 'BC' 
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


with open(INPUT_FILE, 'r') as infile:
    for i, line in enumerate(infile):
        if i%10 == 0:
            print(i)
        fen = line.strip()
        # Get transformer's best move
        board = chess.Board(fen)

        best_move_transformer = get_transformer_move(fen, neural_engine, win_probab=False)
        #Count if the movve is legal
        if board.is_legal(best_move_transformer):
            count_legal += 1
        
        # Get Stockfish's top 5 moves (as UCI strings)
        stockfish_moves_1 = get_stockfish_moves(fen, STOCKFISH_DEPTH, STOCKFISH_TIME, 1, win_probab=False)
        stockfish_moves_5 = get_stockfish_moves(fen, STOCKFISH_DEPTH, STOCKFISH_TIME, 5, win_probab=False)
        stockfish_moves_3 = get_stockfish_moves(fen, STOCKFISH_DEPTH, STOCKFISH_TIME, 3, win_probab=False)
        stockfish_moves_10 = get_stockfish_moves(fen, STOCKFISH_DEPTH, STOCKFISH_TIME, 10, win_probab=False)
        
        
        
        # Check if transformer's move is in Stockfish's top 5
        in_top1 = best_move_transformer.uci() in stockfish_moves_1
        if in_top1:
            count_top1 += 1
        in_top3 = best_move_transformer.uci() in stockfish_moves_3
        if in_top3:
            count_top3 += 1
        in_top5 = best_move_transformer.uci() in stockfish_moves_5
        if in_top5:
            count_top5 += 1
        in_top10 = best_move_transformer.uci() in stockfish_moves_10
        if in_top10:
            count_top10 += 1
        
    print(i+1)
    print(count_legal)
    
    print(count_top1)
    print(count_top3)
    print(count_top5)
    print(count_top10)
