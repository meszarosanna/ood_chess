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
    #s_move = chess.Move.from_uci(s_move)
    t_move = chess.Move.from_uci(t_move)

    markers = [
    chess.svg.SquareMarker(
        square=chess.E4,   # square to mark
        color="red",       # border color
        stroke_width=3,    # border thickness
        fill_opacity=0.0   # no fill
    )
]
    
    # 3. Generate SVG highlighting the move (from and to squares in red)
    svg = chess.svg.board(
        board,
        fill={t_move.from_square: "#02189380", t_move.to_square: "#02189380"}, # s_move.from_square: "#99062380", s_move.to_square: "#99062380", 
        arrows=[chess.svg.Arrow(t_move.from_square, t_move.to_square, color="#02189380")], #chess.svg.Arrow(s_move.from_square, s_move.to_square, color="#99062380"), 
        size=size,
        markers=markers
    )

    
    # 5. Save SVG
    with open(filename, "w") as f:
        f.write(svg)
    
    print(f"Board image saved to {filename}")

# Example usage:
#fen = "1Q3rk1/r7/1R6/3qp2p/3p2pn/1Q1P2p1/4PP2/5RK1 b - - 1 32"
#t_move = "d5g2"
#fen = "4Q3/6pk/2p3p1/8/P4QKP/5PB1/5bq1/3q4 w - - 0 40"
#t_move = "g3f2"
#fen = "r2N3k/3Q2bp/6p1/5p2/8/5R1P/1q4P1/2q1R2K w - - 0 43"
#t_move = "e1e8"
#fen = "1B6/6pk/1n5p/8/4Rp2/5P2/P4BPP/6K1 w - - 1 44"
#fen = "8/8/4B3/8/6b1/4QB1k/1PP3p1/2K5 w - - 0 40"
#t_move = "f2b6"
#t_move = "f3g4"
#fen = "6Q1/p7/3Q4/P7/4pprk/8/4KP1P/8 w - - 2 41"
fen = "nnrkbbrq/pppppppp/8/8/8/8/PPPPPPPP/NNRKBBRQ w - - 0 1"
#fen = "8/8/4p3/pb1bK3/5R2/6k1/1R6/8 b - - 1 66"
t_move = "a1b3"
s_move = "d2d4"
filename="starting_board.svg"
generate_board_image(fen, s_move, t_move, filename)
