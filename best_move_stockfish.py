#!/usr/bin/env python3

import chess
import chess.engine

def get_best_move(fen_string, depth, time, multipv, win_probab):
    """Get the best move for a given FEN position."""
    
    # Create a board from the FEN string
    board = chess.Board(fen_string)

        
    # Initialize Stockfish engine
    with chess.engine.SimpleEngine.popen_uci("./Stockfish/src/stockfish") as engine:
        if win_probab:
            # Configure engine to show WDL
            engine.configure({"UCI_ShowWDL": "true"})
        
            # Get all legal moves
            legal_moves = list(board.legal_moves)
            move_scores = []
        
            # Analyze each legal move
            for move in legal_moves:
                # Make the move
                board.push(move)
            
                # Analyze the resulting position
                result = engine.analyse(board, chess.engine.Limit(depth=depth, time=time))
            
                # Get WDL information
                wdl = result.get("wdl", None)
                if wdl:
                    win, draw, loss = wdl
                    # Convert to percentages
                    win_pct = win / 10.0
                    draw_pct = draw / 10.0
                    loss_pct = loss / 10.0
                
                    # Store move and its evaluation
                    move_scores.append({
                        'move': move,
                        'win_pct': loss_pct,
                        'draw_pct': draw_pct,
                        'loss_pct': win_pct,
                        'score': result.get("score", None)
                    })
            
                # Undo the move
                board.pop()
        
            # Sort moves by win percentage
            move_scores.sort(key=lambda x: x['win_pct'], reverse = True)
        
            # Print results
            print("\nAll legal moves sorted by win probability:")
            print("Move\tWin%\tDraw%\tLoss%\tCentipawns")
            print("-" * 50)
            for move_data in move_scores:
                move = move_data['move']
                score = move_data['score']
                color = board.turn
                score_str = f"{score.white().score() if color == chess.WHITE else score.black().score()}"
                print(f"{move}\t{move_data['win_pct']:.1f}%\t{move_data['draw_pct']:.1f}%\t{move_data['loss_pct']:.1f}%\t{score_str}")
        else:
            # Get top multipv moves
            result = engine.analyse(board, chess.engine.Limit(depth=depth, time=time), multipv=multipv)
            #score = [result[i]["score"] for i in range(multipv)]
            best_move = [result[i]["pv"][0].uci() for i in range(multipv)] # result is a dictionary with keys "pv": list of moves best moves; "score": (centipawns) evaluation of the position; "depth": how deep the engine searched            
            # Print the best move in algebraic notation
            return best_move
        
            

if __name__ == "__main__":
    # The FEN string for the position you want to analyze
    fen = "rnbqkbnr/pppppppp/7r/8/8/7Q/PPPPPPPP/RNBQKBNN b" # KQkq - 0 1"  # Starting position
    #fen = "rnbqkbnr/pppppppp/8/8/8/8/QQQQQQQQ/RNBQKBNR w" # KQkq - 0 1"  # Starting position
    #fen = "rnkqkbnr/pppppQpp/8/8/8/8/PPPPPQPP/RNBQKBNR b"
    
    
    depth = None      # None means unlimited depth
    time = 0.5       # Time limit in seconds (originally, it was 0.05)
    multipv = 5    # Number of best moves to return
    time *= multipv
    win_probab = True # !!!Does not always reflect well the quality of the move

    best_move = get_best_move(fen, depth, time, multipv, win_probab) 
    print(f"Best move: {best_move}")