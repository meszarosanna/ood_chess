#Copyright 2025 Anna Meszaros

"""Evaluate the trained transformer model whether it predicts legal moves and
whether its moves are in Stockfish's top1, top3, top5, top10 moves."""

from collections.abc import Sequence
import chess
import pandas as pd
from tqdm import tqdm
from absl import app
from absl import flags
import itertools
import math
import numpy as np
from searchless_chess.src.engines import constants
from searchless_chess.src.bagz import BagFileReader
from searchless_chess.src.constants import CODERS
from scipy.stats import entropy


#Define engines
MODEL_NAME = 'BC_270M' 
neural_engine = constants.ENGINE_BUILDERS[MODEL_NAME]()
stockfish_engine_sort = constants.ENGINE_BUILDERS['stockfish_sort']()
stockfish_engine_top1 = constants.ENGINE_BUILDERS['stockfish']()
stockfish_engine_top3 = constants.ENGINE_BUILDERS['stockfish_top3']()
stockfish_engine_top5 = constants.ENGINE_BUILDERS['stockfish_top5']()
stockfish_engine_top10 = constants.ENGINE_BUILDERS['stockfish_top10']()

#Define the dataset
_input_file = flags.DEFINE_string(
    name='input_file_name',
    default=None,
    help='The name of the dataset in the "datasets" folder to evaluate on.',
    required=True,
)
    

#Evaluation counters
count_legal = 0 # Counter for legal moves
count_top1 = 0 # Counter for Stockfish top1
count_top3 = 0 # Counter for Stockfish top3
count_top5 = 0 # Counter for Stockfish top5
count_top10 = 0 # Counter for Stockfish top10
count_entire = 0 # Counter for entire sequence match
count_stockfish_error_3 = 0  # Counter for IndexError
count_stockfish_error_5 = 0  # Counter for IndexError
count_stockfish_error_10 = 0  # Counter for IndexError

#Further optional counters
count_move = 0 # Counter for equal move
transformer_entropies = [] # List to store transformer entropies
stockfish_entropies = [] # List to store Stockfish entropies
list_win = [] # List to store win probabilities
list_num =[] # List to store number of moves within 0.1 win probability of the best move


def eval_entire_sequence(board, moves):
    entire_sequence = False
    for move_idx, move in enumerate(moves):
        if move_idx % 2 == 0:
            predicted_move = neural_engine.play(board).uci()
            # Lichess puzzles consider all mate-in-1 moves as correct, so we need to
            # check if the `predicted_move` results in a checkmate if it differs from
            # the solution.
            if move != predicted_move:
                if board.is_legal(chess.Move.from_uci(predicted_move)):
                    board.push(chess.Move.from_uci(predicted_move))
                    entire_sequence = board.is_checkmate()
                    board.pop()
                    break
                else:
                    entire_sequence = False
                    break
        board.push(chess.Move.from_uci(move))
    entire_sequence = True
    if entire_sequence:
        global count_entire
        count_entire += 1

def evalulation(board, t_move):
    # Get transformer's best move
    best_move_transformer = t_move

    #Check if the move is legal
    legal = board.is_legal(best_move_transformer)
    if legal:
        global count_legal
        count_legal += 1
    
    #Check if the transformer's move is in Stockfish's top1,3,5,10
    stockfish_moves_1 = stockfish_engine_top1.play(board).uci()
    in_top1 = (best_move_transformer.uci() == stockfish_moves_1)
    if in_top1:
        global count_top1
        count_top1 += 1
    try:
        result = stockfish_engine_top3.analyse(board)
        stockfish_moves_3 = [result[i]["pv"][0].uci() for i in range(3)]
        in_top3 = (best_move_transformer.uci() in stockfish_moves_3)
        if in_top3:
            global count_top3
            count_top3 += 1
    except IndexError:
        global count_stockfish_error_3
        count_stockfish_error_3 += 1
        in_top3 = None
    try:
        result = stockfish_engine_top5.analyse(board)
        stockfish_moves_5 = [result[i]["pv"][0].uci() for i in range(5)]
        in_top5 = (best_move_transformer.uci() in stockfish_moves_5)
        if in_top5:
            global count_top5
            count_top5 += 1
    except IndexError:
        global count_stockfish_error_5
        count_stockfish_error_5 += 1
        in_top5 = None
    try:
        result = stockfish_engine_top10.analyse(board)
        stockfish_moves_10 = [result[i]["pv"][0].uci() for i in range(10)]
        in_top10 = (best_move_transformer.uci() in stockfish_moves_10)
        if in_top10:
            global count_top10
            count_top10 += 1
    except IndexError:
        global count_stockfish_error_10
        count_stockfish_error_10 += 1
        in_top10 = None
    
def equal_move(move, t_move):
    best_move_transformer = t_move
    equal_move = best_move_transformer.uci() == move
    if equal_move:
        global count_move
        count_move += 1
        
def entropy_transformer(board):
    """Calculate the entropy of the neural engine's move probabilities."""
    total_log_probs = neural_engine.analyse(board)
    probs = np.exp(np.array(total_log_probs))
    probs = probs[probs > 0.1]

    ent = entropy(probs, base=math.e)
    n = len(probs)
    if n <= 1:
        H = 0.0
    else:
        H =  ent / np.log(n)
    global transformer_entropies
    transformer_entropies.append(H)

def entropy_stockfish(board):
    """Calculate the entropy of Stockfish's move probabilities."""
    # Get all legal moves
    move_scores = []
    legal_moves = list(board.legal_moves)
        
    # Analyze each legal move
    for move in legal_moves:
        board.push(move)
        result = stockfish_engine_sort.analyse(board)
        # Get WDL information
        score = result.get("score")
        if score.is_mate():
            win_prob = 1
        else:
            win_prob = 1-1/(1+math.exp(-0.00368208*score.pov(board.turn).score()))
        move_scores.append({
                'move': move.uci(),
                'win_pct': win_prob
                    })
        # Undo the move
        board.pop()
    
    win_probs = [score['win_pct'] for score in move_scores]
    num = len([e for e in win_probs if e > max(win_probs) - 0.1])
    global list_num
    list_num.append(num)
    probs = [float(i)/sum(win_probs) for i in win_probs]
    ent = entropy(probs, base=math.e)
    n = len(probs)
    if n <= 1:
        H = 0.0
    else:
        H =  ent / np.log(n)
    global stockfish_entropies
    stockfish_entropies.append(H)

def list_win_prob(board): 
    """Calculate the win probability of the board."""
    result = stockfish_engine_sort.analyse(board)
    score = result.get("score")
    if score.is_mate():
        win_prob = 1
    else:
        win_prob = 1-1/(1+math.exp(-0.00368208*score.pov(board.turn).score()))
    global list_win
    list_win.append(win_prob)




def main(argv: Sequence[str]) -> None:

    #Define the number of puzzles
    INPUT_FILE = _input_file.value
    if INPUT_FILE == "chess960_puzzles.csv":
        _NUM_PUZZLES = 959
    else:
        _NUM_PUZZLES = 10

    #Read the dataset and evaluate
    if INPUT_FILE == "filtered_behavioral_cloning_test_data.bag":
        reader = BagFileReader("datasets/" + INPUT_FILE)
        for record in tqdm(itertools.islice(reader, _NUM_PUZZLES), total=_NUM_PUZZLES):
            fen, move  = CODERS['behavioral_cloning'].decode(record)
            board = chess.Board(fen)
            t_move = neural_engine.play(board)
            evalulation(board, t_move)
    elif INPUT_FILE in ["id_puzzles.csv", "ood_puzzles.csv"]:
        puzzles = pd.read_csv("datasets/" + INPUT_FILE, nrows=_NUM_PUZZLES)
        for puzzle_id, puzzle in tqdm(puzzles.iterrows(), total=_NUM_PUZZLES):
            fen = puzzle['FEN']
            moves = puzzle['Moves'].split(' ')
            board = chess.Board(fen)
            board.push(chess.Move.from_uci(puzzle["Moves"].split(' ')[0]))
            t_move = neural_engine.play(board)
            evalulation(board), t_move
            eval_entire_sequence(board.copy(), moves[1:])
    elif INPUT_FILE in ["all_ordering_puzzles.csv", "chess960_puzzles.csv", "knights_and_rooks.csv", "more_pieces_puzzles.csv", "same_color_puzzles.csv"]:
        puzzles = pd.read_csv("datasets/" + INPUT_FILE, nrows=_NUM_PUZZLES)
        for puzzle_id, puzzle in tqdm(puzzles.iterrows(), total=_NUM_PUZZLES):
            fen = puzzle['FEN']
            board = chess.Board(fen)
            t_move = neural_engine.play(board)
            evalulation(board, t_move)
    else:
        raise ValueError(f"Unknown input file: {INPUT_FILE}")


    #Print the results
    print(f"When predicting the next move for {INPUT_FILE} consists of {_NUM_PUZZLES} puzzles:")
    print(f'{count_legal} are legal')
    print(f'{count_top1} are in top1 of Stockfish')
    print(f'{count_top3} are in top3 of Stockfish')
    print(f'{count_top5} are in top5 of Stockfish')
    print(f'{count_top10} are in top10 of Stockfish')
    print(f'{count_entire} are matching the entire sequence of the solution')

    print("And Stockfish coundn't give")
    print(f' top3 moves in {count_stockfish_error_3} cases')
    print(f' top5 moves in {count_stockfish_error_5} cases')
    print(f' top10 moves in {count_stockfish_error_10} cases')
    print(f"therefore the percentages are calculated over {(_NUM_PUZZLES - count_stockfish_error_3)} for top3, {(_NUM_PUZZLES - count_stockfish_error_5)} for top5, {(_NUM_PUZZLES - count_stockfish_error_10)} for top10:")
    print(f"Legal: {count_legal/_NUM_PUZZLES*100:.2f}%")
    print(f"Top1: {count_top1/(_NUM_PUZZLES)*100:.2f}%")
    print(f"Top3: {count_top3/(_NUM_PUZZLES - count_stockfish_error_3)*100:.2f}%")
    print(f"Top5: {count_top5/(_NUM_PUZZLES - count_stockfish_error_5)*100:.2f}%")
    print(f"Top10: {count_top10/(_NUM_PUZZLES - count_stockfish_error_10)*100:.2f}%")
    print(f"Entire sequence: {count_entire/_NUM_PUZZLES*100:.2f}%")

    stockfish_engine_sort.__del__()
    stockfish_engine_top1.__del__()
    stockfish_engine_top3.__del__()
    stockfish_engine_top5.__del__()
    stockfish_engine_top10.__del__()
        

if __name__ == '__main__':
  app.run(main)
