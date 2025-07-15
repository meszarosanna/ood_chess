import chess
import os
from searchless_chess.src import utils
from searchless_chess.src.engines import engine
import numpy as np
from load_transformer import load_transformer_model

#to get rid of the warnings: W external/xla/xla/service/gpu/autotuning/dot_search_space.cc:200] All configs were filtered out because none of them sufficiently match the hints. Maybe the hints set does not contain a good representative set of valid configs?Working around this by using the full hints set instead.
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"


def get_best_move(fen_string, neural_engine, win_probab=False):
    """Get the best move for a given FEN position."""
    
    # Create a board from the FEN string
    board = chess.Board(fen_string)

    #Load the transformer model
    #neural_engine = load_transformer_model(model_name)
        
    if win_probab:
        # Get move probabilities for all legal moves
        results = neural_engine.analyse(board)
        sorted_legal_moves = engine.get_ordered_legal_moves(board)

        if model_name == "BC":
            legal_win_probs = np.exp(results['legal_log_probs'])
            total_win_probs = np.exp(results['total_log_probs'])

            all_moves = [utils.ACTION_TO_MOVE[i] for i in range(len(utils.ACTION_TO_MOVE))]
            sorted_legal_moves = engine.get_ordered_legal_moves(board)
            
            #Print win probabilities for all moves
            print(f'Move win probabilities:')
            for idx, i in enumerate(np.argsort(total_win_probs)[::-1]):
                if idx<100:
                    print(f'  {all_moves[i]} -> {100*total_win_probs[i]}%, legal:{all_moves[i] in [x.uci() for x in sorted_legal_moves]}')



        if model_name == "270M":
            l_win_probs = np.exp(results['legal_log_probs'])
            t_win_probs = np.exp(results['total_log_probs'])

            return_buckets_values = neural_engine._return_buckets_values
             # Compute win probabilities for each move
            total_win_probs = []
            for i, move_probs in enumerate(t_win_probs):
                win_prob = np.inner(move_probs, return_buckets_values)
                total_win_probs.append(win_prob)
            total_win_probs = np.array(total_win_probs)


            # Compute win probabilities for each legal move
            legal_win_probs = []
            for i, move_probs in enumerate(l_win_probs):
                win_prob = np.inner(move_probs, return_buckets_values)
                legal_win_probs.append(win_prob)
            legal_win_probs = np.array(legal_win_probs)

            # Print win probabilities for all/legal moves

            print(f'Legal move win probabilities:')
            for i in np.argsort(legal_win_probs)[::-1]:
                print(f'  {sorted_legal_moves[i].uci()} -> {100*legal_win_probs[i]}%')


            all_moves = [utils.ACTION_TO_MOVE[i] for i in range(len(utils.ACTION_TO_MOVE))]

            print(f'All move win probabilities:')
            for idx, i in enumerate(np.argsort(total_win_probs)[::-1]):
                if idx <100:
                    print(f'  {all_moves[i]} -> {100*total_win_probs[i]}%, legal: {all_moves[i] in [x.uci() for x in sorted_legal_moves]}')



        
    best_move = neural_engine.play(board)
    print(f'Best move: {best_move}')
    return best_move
    
        

if __name__ == "__main__":

    model_name = "270M"
    model = load_transformer_model(model_name)
    # The FEN string for the position you want to analyze
    #fen = "rnbqkbnr/pppppppp/7r/8/8/7Q/PPPPPPPP/RNBQKBNN b" # KQkq - 0 1"  # Starting position
    #fen = "rnbqkbnr/pppppppp/8/8/8/8/QQQQQQQQ/RNBQKBNR w"
    #fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"
    #fen = "4r1k1/2p1qpp1/3p4/1p1P2PQ/1P6/3R3P/2PBrb2/5RK1 w - - 7 27"
    #fen = "r4k2/1pp2Bp1/5n1p/1q2N3/3P4/7P/5PP1/4Q1K1 w - - 3 31"
    #fen = "3r1rk1/4Qppp/8/1ppb4/2Pn1B1n/2N3P1/PP3P2/R2R1K2 b - - 0 21"
    #fen = "r3r1k1/1Q3Rpp/8/pP6/2q5/7P/3R2P1/7K b - - 0 30"
    #fen = "r4r2/ppp1qpk1/3p4/2bnp3/2B1P1pR/3P2B1/PPPK1P2/7R w - - 0 19"
    #fen = "8/p5k1/1pp3p1/4pp1p/5P1P/2P3P1/PP4K1/8 w - - 0 36"
    #fen = "1kr5/p3b3/p1p1p3/B1P1Np1p/3P1P1q/1P5b/1PR2RPP/6K1 w - - 0 28"
    #fen = "2k5/1pp1r2Q/pr2p3/3p1p2/q2b4/P2NP3/1PP3PP/K3RR2 w - - 0 26"
    #fen = "r3kb1r/ppp2ppp/8/2p1Pb2/3Pn2q/3B1P2/PPP1K1PP/RNBQ3R b kq - 2 9"
    fen = "2n5/p1k5/7p/5Kp1/P1B2pP1/5P2/7P/8 b - - 1 46"
    board = chess.Board(fen)
    
    win_probab=False # Lists the win probability for all legal moves
    
    get_best_move(fen, model, win_probab)
    