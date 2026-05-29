#Copyright 2025 Anna Meszaros

"""Creates chess puzzles where a king can be checkmated by a knight while being blocked by rooks."""

import chess
import pandas as pd
import random
from tqdm import tqdm

NUM_PUZZLES = 1000

for _ in tqdm(range(NUM_PUZZLES)):
    board = chess.BaseBoard.empty()
    moves = ""
    #Place the king
    square = random.randint(0, 63)
    board.set_piece_at(square, chess.Piece.from_symbol('k'))

    #Place rooks to block king movement
    if square < 16:
        if square % 8 == 0:
            board.set_piece_at(57, chess.Piece.from_symbol('R'))
        elif square % 8 == 7:
            board.set_piece_at(62, chess.Piece.from_symbol('R'))
        else:
            board.set_piece_at(56+square%8-1, chess.Piece.from_symbol('R'))
            board.set_piece_at(56+square%8+1, chess.Piece.from_symbol('R'))
    elif square > 47:
        if square % 8 == 0:
            board.set_piece_at(1, chess.Piece.from_symbol('R'))
        elif square % 8 == 7:
            board.set_piece_at(6, chess.Piece.from_symbol('R'))
        else:
            board.set_piece_at(square%8-1, chess.Piece.from_symbol('R'))
            board.set_piece_at(square%8+1, chess.Piece.from_symbol('R'))
    else:
        if square % 8 == 0:
            board.set_piece_at(random.choice([1, 57]), chess.Piece.from_symbol('R'))
        elif square % 8 == 7:
            board.set_piece_at(random.choice([6, 62]), chess.Piece.from_symbol('R'))
        else:
            board.set_piece_at(random.choice([56+square%8-1, square%8-1]), chess.Piece.from_symbol('R'))
            board.set_piece_at(random.choice([56+square%8+1, square%8+1]), chess.Piece.from_symbol('R'))
    if square%8 == 0 or square%8 == 1:
        if square <=7:
            board.set_piece_at(15, chess.Piece.from_symbol('R'))
        elif square >= 56:
            board.set_piece_at(55, chess.Piece.from_symbol('R'))
        else:
            board.set_piece_at(square-square%8+15, chess.Piece.from_symbol('R'))
            board.set_piece_at(square-square%8-1, chess.Piece.from_symbol('R'))
    elif square%8 == 6 or square%8 == 7:
        if square <=7:
            board.set_piece_at(8, chess.Piece.from_symbol('R'))
        elif square >= 56:
            board.set_piece_at(48, chess.Piece.from_symbol('R'))
        else:
            board.set_piece_at(square-square%8+8, chess.Piece.from_symbol('R'))
            board.set_piece_at(square-square%8-8, chess.Piece.from_symbol('R'))
    else:
        if square <=7:
            board.set_piece_at(random.choice([8, 15]), chess.Piece.from_symbol('R'))
        elif square >= 56:
            board.set_piece_at(random.choice([48, 55]), chess.Piece.from_symbol('R'))
        else:
            board.set_piece_at(random.choice([square-square%8+8, square-square%8+15]), chess.Piece.from_symbol('R'))
            board.set_piece_at(random.choice([square-square%8-8, square-square%8-1]), chess.Piece.from_symbol('R'))

    #Place 1 knight to attack the king 
    if square % 8 == 0:
        knights = [square-15, square+17, square-6, square+10]
    elif square % 8 == 1:
        knights = [square-15, square+17, square-6, square+10, square+15, square-17]
    elif square % 8 == 6:
        knights = [square-17, square+15, square-10, square+6, square+17, square-15]
    elif square % 8 == 7:
        knights = [square-17, square+15, square-10, square+6]
    else:
        knights = [square-17, square+15, square-10, square+6, square-15, square+17, square-6, square+10]
    knights = [e for e in knights if e >= 0 and e < 64 and e != square]

    knight_set = False
    empty_squares = [e for e in range(64) if board.piece_at(e) is None and e not in knights]
    while not knight_set:
        n = random.choice(empty_squares)
        board.set_piece_at(n, chess.Piece.from_symbol('N'))
        if board.attacks(n) & set(knights):
            knight_set = True
            for m in board.attacks(n) & set(knights):
                moves += chess.SQUARE_NAMES[n] + chess.SQUARE_NAMES[m] + ' '
        else:
            board.remove_piece_at(n)

    #Place additional knights
    number_of_knights = random.randint(2, 14)
    for _ in range(number_of_knights):
        knight_set = False
        empty_squares = [e for e in range(64) if board.piece_at(e) is None and e not in knights]
        while not knight_set:
            n = random.choice(empty_squares)
            board.set_piece_at(n, chess.Piece.from_symbol('N'))
            if not (board.attacks(n) & set(knights)):
                knight_set = True
            else:
                board.remove_piece_at(n)


     
    puzzle_fen = board.board_fen()

    filename = "knights_and_rooks.csv"
    with open(filename, "a") as outfile:
        outfile.write(puzzle_fen + ','+ moves + '\n')


# Filter: keep only puzzles with exactly one mating move
def get_mate_moves(row):
    board = chess.Board(row['FEN'])
    mate_moves = []
    for m in row['Moves'].strip().split():
        try:
            board2 = board.copy()
            board2.push(chess.Move.from_uci(m))
            if board2.is_checkmate():
                mate_moves.append(m)
        except Exception:
            pass
    return mate_moves

df = pd.read_csv('knights_and_rooks.csv', names=['FEN', 'Moves'])
df['mate_moves'] = df.apply(get_mate_moves, axis=1)
filtered = df[df['mate_moves'].apply(len) == 1].copy()
filtered['best_move'] = filtered['mate_moves'].apply(lambda x: x[0])
filtered[['FEN', 'Moves', 'best_move']].to_csv('knights_and_rooks_single_mate.csv', index=False)
print(f'Saved {len(filtered)} / {len(df)} puzzles with a single mating move.')

