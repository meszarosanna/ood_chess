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




MODEL_NAME = 'BC_270M_64' 
#neural_engine = load_transformer_model(MODEL_NAME)
neural_engine = constants.ENGINE_BUILDERS[MODEL_NAME]()
stockfish_engine_sort = constants.ENGINE_BUILDERS['stockfish_sort']()
stockfish_engine_top1 = constants.ENGINE_BUILDERS['stockfish']()
stockfish_engine_top3 = constants.ENGINE_BUILDERS['stockfish_top3']()
stockfish_engine_top5 = constants.ENGINE_BUILDERS['stockfish_top5']()
stockfish_engine_top10 = constants.ENGINE_BUILDERS['stockfish_top10']()

# Stockfish parameters
#STOCKFISH_DEPTH = None
#STOCKFISH_TIME = 0.5

count_legal = 0
count_top5 = 0
count_top3 = 0
count_top1 = 0
count_top1v2 = 0
count_movev2 = 0
count_movev3 = 0
count_top10 = 0
count_entire = 0
count_move = 0
count_mate = 0
count_pinned = 0
count_wdl_error = 0
count_stockfish_error_3 = 0  # Counter for IndexError
count_stockfish_error_5 = 0  # Counter for IndexError
count_stockfish_error_10 = 0  # Counter for IndexError
entropies = []

INPUT_FILE =  'ood_puzzles.csv' #'searchless_chess/data/test/filtered_behavioral_cloning_data.bag'
_NUM_PUZZLES = 1000
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
#for record in tqdm(itertools.islice(reader, _NUM_PUZZLES), total=_NUM_PUZZLES):
#for record in tqdm(reader, total=num_of_rec):
    #fen, move  = CODERS['behavioral_cloning'].decode(record)
    fen = puzzle['FEN']
    move = puzzle['Moves'].split(' ')[1]
    board = chess.Board(fen)


    # Get transformer's best move
    board.push(chess.Move.from_uci(puzzle["Moves"].split(' ')[0]))

    #pinned_squares = []
    #for square in chess.SQUARES:
    #    piece = board.piece_at(square)
    #    if piece and piece.color == board.turn and board.is_pinned(piece.color, square):
    #        pinned_squares.append(square)

    #if len(pinned_squares) > 0:
    #    count_pinned += 1
    
    best_move_transformer = neural_engine.play(board)
    

    #Check if the move is legal
    legal = board.is_legal(best_move_transformer)
    if legal:
        count_legal += 1
    
    #    board.push(best_move_transformer)
    #    if board.is_checkmate():
    #        count_mate += 1
    #    board.pop()

    #list_of_moves = move.split(' ')
    #print(list_of_moves)
    #if best_move_transformer.uci() in list_of_moves:
    #    count_move += 1

    # Check if transformer's move is in Stockfish's top 1,3,5,10
        
    #stockfish_moves_1 = get_stockfish_moves(fen, STOCKFISH_DEPTH, STOCKFISH_TIME, 1, win_probab=False)
    #try:
    #    result = stockfish_engine_top3.analyse(board)
    #    stockfish_moves_3 = [result[i]["pv"][0].uci() for i in range(3)]
    #    #stockfish_moves_5 = get_stockfish_moves(fen, STOCKFISH
    #    in_top3 = (best_move_transformer.uci() in stockfish_moves_3)
    #    if in_top3:
    #        count_top3 += 1
    #except IndexError:
    #    count_stockfish_error_3 += 1

    #Check if matches the entire solution
    #entire_sequence = eval_entire_sequence(board.copy())
    #if entire_sequence:
    #    count_entire += 1

    #equal_move = best_move_transformer.uci() == move
    #if equal_move:
    #    count_move += 1
        
    stockfish_moves_1 = stockfish_engine_top1.play(board).uci()
    in_top1 = (best_move_transformer.uci() == stockfish_moves_1)
    if in_top1:
        count_top1 += 1

        
    # Get all legal moves
    #legal_moves = list(board.legal_moves)
    #move_scores = []
        
    # Analyze each legal move
    #for move in legal_moves:
        # Make the move
    #    board.push(move)
            
        # Analyze the resulting position
    #    result = stockfish_engine_sort.analyse(board)
            
        # Get WDL information
    #    score = result.get("score")
    #    if score.is_mate():
    #        win_prob = 1
    #    else:
    #        win_prob = 1-1/(1+math.exp(-0.00368208*score.pov(board.turn).score()))
    #    move_scores.append({
    #            'move': move,
    #            'win_pct': win_prob
    #                })
        # Undo the move
    #    board.pop()
    
    #win_probs = [score['win_pct'] for score in move_scores]
    #probs = [float(i)/sum(win_probs) for i in win_probs]
    #ent = entropy(probs, base=math.e)
    #n = len(probs)
    #if n <= 1:
    #    H = 0.0
    #else:
    #    H =  ent / np.log(n)
    #entropies.append(H)
        
    # Sort moves by win percentage
    #move_scores.sort(key=lambda x: x['win_pct'], reverse = True)
    #stockfish_sorted = move_scores[0]['move'].uci()
    #if best_move_transformer.uci() == stockfish_sorted:
    #    count_top1v2 += 1
    #if stockfish_sorted == move:
    #    count_movev2 += 1
    #if stockfish_sorted == stockfish_moves_1:
    #    count_movev3 += 1

    #try:
    #    result = stockfish_engine_top3.analyse(board)
    #    stockfish_moves_3 = [result[i]["pv"][0].uci() for i in range(3)]
    #    in_top3 = (best_move_transformer.uci() in stockfish_moves_3)
    #    if in_top3:
    #        count_top3 += 1
    #except IndexError:
    #    count_stockfish_error_3 += 1
    #    in_top3 = None
    #try:
    #    result = stockfish_engine_top5.analyse(board)
    #    stockfish_moves_5 = [result[i]["pv"][0].uci() for i in range(5)]
    #    in_top5 = (best_move_transformer.uci() in stockfish_moves_5)
    #    if in_top5:
    #        count_top5 += 1
    #except IndexError:
    #    count_stockfish_error_5 += 1
    #    in_top5 = None
    #try:
    #    result = stockfish_engine_top10.analyse(board)
    #    stockfish_moves_10 = [result[i]["pv"][0].uci() for i in range(10)]
        #stockfish_moves_3 = stockfish_moves_10[:3]
        #stockfish_moves_5 = stockfish_moves_10[:5]
        #in_top3 = (best_move_transformer.uci() in stockfish_moves_3)
        #in_top5 = (best_move_transformer.uci() in stockfish_moves_5)
        #if in_top3:
        #    count_top3 += 1
        #if in_top5:
        #    count_top5 += 1
    #    in_top10 = (best_move_transformer.uci() in stockfish_moves_10)
    #    if in_top10:
    #        count_top10 += 1
    #except IndexError:
    #    count_stockfish_error_10 += 1
    #    in_top10 = None
    #try:
    #    result = stockfish_engine_top10.analyse(board)
    #    stockfish_moves_10 = [result[i]["pv"][0].uci() for i in range(10)]
    #    stockfish_moves_1v2 = stockfish_moves_10[0]
    #    stockfish_moves_3 = stockfish_moves_10[:3]
    #    stockfish_moves_5 = stockfish_moves_10[:5]
    #    in_top1v2 = (best_move_transformer.uci() == stockfish_moves_1)
    #    in_top3 = best_move_transformer.uci() in stockfish_moves_3
    #    in_top5 = best_move_transformer.uci() in stockfish_moves_5
    #    in_top10 = best_move_transformer.uci() in stockfish_moves_10
    #    if in_top1v2:
    #        count_top1v2 += 1
    #    if in_top3:
    #        count_top3 += 1
    #    if in_top5:
    #    if in_top10:
    #        count_top5 += 1
    #        count_top10 += 1
    #    with open('log_more_pieces.csv', 'a') as file:
    #        file.write(f"{board.fen()},{move},{best_move_transformer.uci()},{legal},{stockfish_moves_10},{in_top1},{in_top3},{in_top5},{in_top10},{equal_move}\n")
    #except IndexError:
    #    count_stockfish_error_10 += 1
    #    with open('log_more_pieces.csv', 'a') as file:
    #        file.write(f"{board.fen()},{move},{best_move_transformer.uci()},{legal},None,None,None,None,None,{equal_move}\n")
    
    #stockfish_move = stockfish_engine_top1.play(board).uci()
    #if best_move_transformer.uci() == stockfish_move:
    #    count_top1 += 1
    #if stockfish_moves_1 == move:
    #    count_move += 1
    #if stockfish_moves_1v2 == move:
    #    count_movev2 += 1
    
    
#print(sum(entropies)/len(entropies))
print(f"When predicting the next move, from {_NUM_PUZZLES} puzzles:")
#print(f'{count_legal} are legal')
print(f'{count_top1} are in top1 of Stockfish')
print(f'{count_top3} are in top3 of Stockfish')
print(f'{count_top5} are in top5 of Stockfish')
print(f'{count_top10} are in top10 of Stockfish')
#print(f'{count_entire} are matching the entire sequence of the solution')
print(f'{count_move} are equal to the given move')
print(f'{count_movev2} are equal to the given move (Stockfish top1 sorted by WDL)')
print(f'{count_movev3} are equal to Stockfish top1 (Stockfish top1 sorted by WDL)')
print(f'{count_top1v2} are in top1 of Stockfish (sorted by WDL)')
print(f'{count_wdl_error} WDL errors')
#print(f'{count_pinned} pinned')

print("And Stockfish coundn't give")
print(f' top10 moves in {count_stockfish_error_10} cases')
print("therefroe")
#print(f"the percentages are calculated over {(_NUM_PUZZLES - count_stockfish_error_10)} for top1,3,5,10")
print(f"the percentages are calculated over {(_NUM_PUZZLES - count_stockfish_error_3)} for top3, {(_NUM_PUZZLES - count_stockfish_error_5)} for top5, {(_NUM_PUZZLES - count_stockfish_error_10)} for top10")
print(f"Legal: {count_legal/_NUM_PUZZLES*100:.2f}%")
print(f"Top1: {count_top1/(_NUM_PUZZLES)*100:.2f}%")
#print(f"Top1v2: {count_top1v2/(_NUM_PUZZLES - count_stockfish_error_10)*100:.2f}%")
print(f"Top3: {count_top3/(_NUM_PUZZLES - count_stockfish_error_3)*100:.2f}%")
print(f"Top5: {count_top5/(_NUM_PUZZLES - count_stockfish_error_5)*100:.2f}%")
print(f"Top10: {count_top10/(_NUM_PUZZLES - count_stockfish_error_10)*100:.2f}%")
print(f"Entire sequence: {count_entire/_NUM_PUZZLES*100:.2f}%")
#print(f"Equal with given move: {count_move/_NUM_PUZZLES*100:.2f}%")
#print(f"Equal with given movev2: {count_movev2/_NUM_PUZZLES*100:.2f}%")
print(f"Resulted in mate: {count_mate/_NUM_PUZZLES*100:.2f}%")


#stockfish_engine_top1.__del__()
#stockfish_engine_top10.__del__()
#stockfish_engine_top3.__del__()
