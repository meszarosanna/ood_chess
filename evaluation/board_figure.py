#Copyright 2025 Anna Meszaros

import chess
import chess.svg
import cairosvg

def generate_board_image(fen, t_move, filename, size=400):
    """
    Generate a chess board image from a FEN string and a move in UCI format (e2e4).
    Highlights the move on the board.
    
    Args:
        fen (str): FEN notation of the position.
        t_move (str): Move in UCI format, e.g., "e2e4".
        filename (str): Output filename for SVG.
        size (int): Size of the board in pixels.
    """
    board = chess.Board(fen)
    t_move = chess.Move.from_uci(t_move)
    
    svg = chess.svg.board(
        board,
        fill={t_move.from_square: "#02189380", t_move.to_square: "#02189380"},
        arrows=[chess.svg.Arrow(t_move.from_square, t_move.to_square, color="#02189380")], 
        size=size,
    )

    cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to=filename)
    

# Example usage:
fen = "1Q3rk1/r7/1R6/3qp2p/3p2pn/1Q1P2p1/4PP2/5RK1 b - - 1 32"
t_move = "d5g2"
filename="board.pdf"
generate_board_image(fen, t_move, filename)
