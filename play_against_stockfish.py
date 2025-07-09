import chess
import chess.engine
import os

def main():
    # Initialize Stockfish engine with full path
    stockfish_path = os.path.join(os.getcwd(), "Stockfish", "src", "stockfish")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    # Initialize the board
    fen = "rnbqkbnr/pppppppr/8/8/8/8/PPPPPPPQ/RNBQKBNN b" # KQkq - 0 1"  # Starting position
    board = chess.Board(fen)
    
    # Main game loop
    while not board.is_game_over():
        # Print current position
        print("\nCurrent position:")
        print(board)
        
        if board.turn == chess.WHITE:
            # Human's turn (White)
            while True:
                try:
                    move_str = input("\nEnter your move (e.g., 'e2e4'): ")
                    move = chess.Move.from_uci(move_str)
                    # Accept any move without validation
                    board.push(move)
                    break
                except ValueError:
                    print("Invalid move format! Use format like 'e2e4'")
        else:
            # Stockfish's turn (Black)
            print("\nStockfish is thinking...")
            result = engine.play(board, chess.engine.Limit(time=2.0))  # 2 seconds thinking time
            best_move = result.move
            print(f"Stockfish plays: {best_move}")
            board.push(best_move)
    
    # Game over
    print("\nGame Over!")
    print(f"Result: {board.result()}")
    
    # Clean up
    engine.quit()

if __name__ == "__main__":
    main() 