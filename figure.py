import chess
import chess.svg

def generate_board_image(fen, s_move, t_move, filename, size=400):
    """
    Generate a chess board image from a FEN string and a move in UCI format (e2e4).
    Highlights the move on the board.
    
    Args:
        fen (str): FEN notation of the position.
        uci_move (str): Move in UCI format, e.g., "e2e4".
        filename (str): Output filename for SVG.
        size (int): Size of the board in pixels.
    """
    # 1. Create the board from FEN
    board = chess.Board(fen)
    
    # 2. Parse UCI move
    s_move = chess.Move.from_uci(s_move)
    t_move = chess.Move.from_uci(t_move)
    
    # 3. Generate SVG highlighting the move (from and to squares in red)
    svg = chess.svg.board(
        board,
        fill={s_move.from_square: "#99062380", s_move.to_square: "#99062380", t_move.from_square: "#02189380", t_move.to_square: "#02189380"},
        arrows=[chess.svg.Arrow(s_move.from_square, s_move.to_square, color="#99062380"), chess.svg.Arrow(t_move.from_square, t_move.to_square, color="#02189380")],
        size=size
    )

    
    # 5. Save SVG
    with open(filename, "w") as f:
        f.write(svg)
    
    print(f"Board image saved to {filename}")

# Example usage:
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
s_move = "d2d4"
t_move = "b1c3"
filename="board1.svg"
generate_board_image(fen, s_move, t_move, filename)
