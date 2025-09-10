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
from best_move_transformer import get_best_move as get_transformer_move
from best_move_transformer import load_transformer_model
from best_move_stockfish import get_best_move as get_stockfish_moves
from searchless_chess.src.engines import constants
from bagz import BagFileReader
from constants import CODERS




MODEL_NAME = 'BC_270M' 
#neural_engine = load_transformer_model(MODEL_NAME)
neural_engine = constants.ENGINE_BUILDERS[MODEL_NAME]()
#stockfish_engine_top1 = constants.ENGINE_BUILDERS['stockfish']()
stockfish_engine_top10 = constants.ENGINE_BUILDERS['stockfish_top10']()

# Stockfish parameters
#STOCKFISH_DEPTH = None
#STOCKFISH_TIME = 0.5

count_legal = 0
count_top5 = 0
count_top3 = 0
count_top1 = 0
count_top10 = 0
count_entire = 0
count_move = 0
count_mate = 0
count_stockfish_error_3 = 0  # Counter for IndexError
count_stockfish_error_5 = 0  # Counter for IndexError
count_stockfish_error_10 = 0  # Counter for IndexError

INPUT_FILE = 'searchless_chess/data/puzzles.csv'
_NUM_PUZZLES = 10
puzzles = pd.read_csv(INPUT_FILE, nrows=_NUM_PUZZLES)

#reader = BagFileReader(INPUT_FILE)
#num_of_rec = len(reader)

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
 

for puzzle_id, puzzle in tqdm(puzzles.iterrows(), total=_NUM_PUZZLES):
#for record in tqdm(itertools.islice(reader, 100), total=100):
#for record in tqdm(reader, total=num_of_rec):
    #fen, move  = CODERS['behavioral_cloning'].decode(record)
    fen = puzzle['FEN']
    move = puzzle['Moves'].split(' ')[1]
    board = chess.Board(fen)

    # Get transformer's best move
    board.push(chess.Move.from_uci(puzzle["Moves"].split(' ')[0]))
    
    best_move_transformer = neural_engine.play(board)

    #Check if the move is legal
    if board.is_legal(best_move_transformer):
        count_legal += 1
        #board.push(best_move_transformer)
        #if board.is_checkmate():
        #    count_mate += 1
        #board.pop()

    #list_of_moves = move.split(' ')
    #print(list_of_moves)
    #if best_move_transformer.uci() in move:
    #    count_move += 1

    # Check if transformer's move is in Stockfish's top 1,3,5,10
        
    #stockfish_moves_1 = get_stockfish_moves(fen, STOCKFISH_DEPTH, STOCKFISH_TIME, 1, win_probab=False)
    #stockfish_moves_1 = stockfish_engine_top1.play(board).uci()
    #in_top1 = (best_move_transformer.uci() == stockfish_moves_1)
    #if in_top1:
    #    count_top1 += 1
        
    
    try:
        result = stockfish_engine_top10.analyse(board)
        stockfish_moves_10 = [result[i]["pv"][0].uci() for i in range(10)]
        stockfish_moves_1 = stockfish_moves_10[0]
        stockfish_moves_3 = stockfish_moves_10[:3]
        stockfish_moves_5 = stockfish_moves_10[:5]
        in_top1 = (best_move_transformer.uci() == stockfish_moves_1)
        in_top3 = best_move_transformer.uci() in stockfish_moves_3
        in_top5 = best_move_transformer.uci() in stockfish_moves_5
        in_top10 = best_move_transformer.uci() in stockfish_moves_10
        if in_top1:
            count_top1 += 1
        if in_top3:
            count_top3 += 1
        if in_top5:
            count_top5 += 1
        if in_top10:
            count_top10 += 1
    except IndexError:
        count_stockfish_error_10 += 1
    
    
    
    #Check if matches the entire solution
    if eval_entire_sequence(board):
        count_entire += 1

    #if stockfish_moves_1 == move:
    #    count_move += 1

print(f"When predicting the next move, from {_NUM_PUZZLES} puzzles:")
print(f'{count_legal} are legal')
print(f'{count_top1} are in top1 of Stockfish')
print(f'{count_top3} are in top3 of Stockfish')
print(f'{count_top5} are in top5 of Stockfish')
print(f'{count_top10} are in top10 of Stockfish')
print(f'{count_entire} are matching the entire sequence of the solution')

print("And Stockfish coundn't give")
print(f' top10 moves in {count_stockfish_error_10} cases')
print("therefroe")
print(f"the percentages are calculated over {(_NUM_PUZZLES - count_stockfish_error_10)} for top1,3,5,10")
print(f"Legal: {count_legal/_NUM_PUZZLES*100:.2f}%")
print(f"Top1: {count_top1/(_NUM_PUZZLES - count_stockfish_error_10)*100:.2f}%")
print(f"Top3: {count_top3/(_NUM_PUZZLES - count_stockfish_error_10)*100:.2f}%")
print(f"Top5: {count_top5/(_NUM_PUZZLES - count_stockfish_error_10)*100:.2f}%")
print(f"Top10: {count_top10/(_NUM_PUZZLES - count_stockfish_error_10)*100:.2f}%")
print(f"Entire sequence: {count_entire/_NUM_PUZZLES*100:.2f}%")
#print(f"Equal with given move: {count_move/_NUM_PUZZLES*100:.2f}%")
#print(f"Resulted in mate: {count_mate/_NUM_PUZZLES*100:.2f}%")


stockfish_engine_top10.__del__()

