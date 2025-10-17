# Copyright 2025 DeepMind Technologies Limited AND Meszaros et al.
#
# The original file was edited by Anna Meszaros to support the turnament of the BC_270M model against stockfishes and fairy-stockfishes
# on variants chess960 and horde, and to count the % of illegal moves and OOD positions played by BC_270M.

"""Launches a tournament between various engines to compute their Elos on different variants."""

from collections.abc import Mapping, Sequence
import copy
import datetime
import itertools
import os
import pandas as pd 

from absl import app
from absl import flags
import chess
import chess.engine
import chess.pgn
import chess.variant
import numpy as np

from searchless_chess.src.engines import constants
from searchless_chess.src.engines import engine
from searchless_chess.src.engines import stockfish_engine


piece_num = { chess.PAWN : 8, chess.ROOK : 2, chess.KNIGHT : 2, chess.BISHOP : 2, chess.QUEEN : 1}

def check_ood(board):
    bool_all_ood = False

    #Check if the number of piece type exceeds the limit
    for piece, num in piece_num.items():
        if (len(board.pieces(piece, chess.WHITE)) > num) or (len(board.pieces(piece, chess.BLACK)) > num):
            bool_all_ood = True

    #Check if there are 2 bishops on the same color
    if len(board.pieces(chess.BISHOP, chess.WHITE)) == 2:
        l = list(board.pieces(chess.BISHOP, chess.WHITE))
        if (chess.square_rank(l[0]) + chess.square_file(l[0]) + chess.square_rank(l[1]) + chess.square_file(l[1]))%2 == 0:
            bool_all_ood = True
    if len(board.pieces(chess.BISHOP, chess.BLACK)) == 2:
        l= list(board.pieces(chess.BISHOP, chess.BLACK))
        if (chess.square_rank(l[0]) + chess.square_file(l[0]) + chess.square_rank(l[1]) + chess.square_file(l[1]))%2 == 0:
            bool_all_ood = True

    return bool_all_ood




os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

_NUM_GAMES = flags.DEFINE_integer(
    name='num_games',
    default=None,
    help='The number of games to play between each pair of engines.',
    required=True,
)
_VARIANT = flags.DEFINE_string(
    name='variant',
    default=None,
    help='Chess variant to play the tournament on: standard, chess960, horde.',
    required=True,
)
  

_MIN_SCORE_TO_STOP = 1300

count_ood = 0
count_move = 0
count_illegal = 0

def _play_game(
    engines: tuple[engine.Engine, engine.Engine],
    engines_names: tuple[str, str],
    white_name: str,
    initial_board: chess.Board | None = None,
) -> chess.pgn.Game:
  """Plays a game of chess between two engines.

  Args:
    engines: The engines to play the game.
    engines_names: The names of the engines.
    white_name: The name of the engine playing white.
    initial_board: The initial board (if None, the standard starting position).

  Returns:
    The game played between the engines.
  """
  if initial_board is None:
    initial_board = chess.Board()
  white_player = engines_names.index(white_name)
  current_player = white_player if initial_board.turn else 1 - white_player
  board = initial_board
  result = None
  print(f'Starting FEN: {board.fen()}')
  #with open('logs.csv', 'a') as file:
  #    file.write(f'Starting FEN: {board.fen()}\n')


  # We use a stockfish engine to evaluate the current board and terminate the
  # game early if the score is high enough (i.e., _MIN_SCORE_TO_STOP).
  if _VARIANT.value == 'standard' or 'chess960':
      _EVAL_STOCKFISH_ENGINE = stockfish_engine.StockfishEngine(limit=chess.engine.Limit(time=0.01))

  while not (
      board.is_game_over()
      or board.can_claim_fifty_moves()
      or board.is_repetition()
  ):
    if engines_names[current_player] == "BC_270M":
      if check_ood(board):
        global count_ood
        count_ood +=1
      best_move = engines[current_player].play(board, legal=False)
      global count_move
      count_move += 1
      if not board.is_legal(best_move):
          best_move = engines[current_player].play(board, legal=True)
          global count_illegal
          count_illegal += 1
    else:
      best_move = engines[current_player].play(board)

    print(f"Move of {engines_names[current_player]}: {best_move}")

    # Push move to the game.
    board.push(best_move)
    current_player = 1 - current_player

    # We analyse the board once the last move is done and pushed.
    if _VARIANT.value == 'standard' or _VARIANT.value == 'chess960':
        info = _EVAL_STOCKFISH_ENGINE.analyse(board)
        score = info['score'].relative
        #with open('delete.csv', 'a') as file:
        #  file.write(f"{score}\n")
        if score.is_mate():
          is_winning = score.mate() > 0
        else:
          is_winning = score.score() > 0
        score_too_high = score.is_mate() or abs(score.score()) > _MIN_SCORE_TO_STOP

        if score_too_high:
          is_white = board.turn == chess.WHITE
          if is_white and is_winning or (not is_white and not is_winning):
            result = '1-0'
          else:
            result = '0-1'
          break
    #with open('delete.csv', 'a') as file:
    #  file.write(f'End FEN: {board.fen()}\n')
    print(f'End FEN: {board.fen()}')

  game = chess.pgn.Game.from_board(board)
  game.headers['Event'] = 'UAIChess'
  game.headers['Date'] = datetime.datetime.today().strftime('%Y.%m.%d')
  game.headers['White'] = white_name
  game.headers['Black'] = engines_names[1 - white_player]
  if result is not None:  # Due to early stopping.
    game.headers['Result'] = result
  else:
    game.headers['Result'] = board.result(claim_draw=True)
  return game


def _run_tournament(
    engines: Mapping[str, engine.Engine],
    opening_boards: Sequence[chess.Board],
) -> Sequence[chess.pgn.Game]:
  """Runs a tournament between engines given openings.

  We play both sides for each opening, and the total number of games played per
  pair is therefore 2 * len(opening_boards).

  Args:
    engines: A mapping from engine names to engines.
    opening_boards: The boards to use as openings.

  Returns:
    The games played between all the engines.
  """
  games = list()

  for engine_name_0, engine_name_1 in itertools.combinations(engines, 2):
    print(f'Playing games between {engine_name_0} and {engine_name_1}')
    with open('delete.csv', 'a') as file:
      file.write(f'Playing games between {engine_name_0} and {engine_name_1}\n')
    engine_0 = engines[engine_name_0]
    engine_1 = engines[engine_name_1]

    for opening_board, white_idx in itertools.product(opening_boards, (0, 1)):
      white_name = (engine_name_0, engine_name_1)[white_idx]
      game = _play_game(
          engines=(engine_0, engine_1),
          engines_names=(engine_name_0, engine_name_1),
          white_name=white_name,
          # Copy as we modify the opening board in the function.
          initial_board=copy.deepcopy(opening_board),
      )
      games.append(game)

  return games


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  

  # To ensure variability in the games we play, we use the openings from the
  # Encyclopedia of Chess Openings.
  openings_path = os.path.join(
      os.getcwd(),
      'searchless_chess/data/eco_openings.pgn',
  )
  opening_boards = list()

  with open(openings_path, 'r') as file:
    while (game := chess.pgn.read_game(file)) is not None:
      opening_boards.append(game.end().board())
  
  # We subsample the openings according to the desired number of games.
  rng = np.random.default_rng(seed=1)
  opening_indices = rng.choice(
      np.arange(len(opening_boards)),
      # Divide by two as we consider both sides per opening (white and black).
      size=_NUM_GAMES.value // 2,
      replace=False,
  )
  opening_boards = list(opening_boards[idx] for idx in opening_indices)


  # Load chess960 openings from a separate file
  chess960_opening_boards = list()
  chess960_start = pd.read_csv('searchless_chess/data/chess960_openings_20steps.csv')
  for id, puzzle in chess960_start.iterrows():
      board = chess.Board(puzzle['FEN'], chess960=True)
      chess960_opening_boards.append(board)

  # We subsample the openings according to the desired number of games.
  rng = np.random.default_rng(seed=1)
  opening_indices = rng.choice(
      np.arange(len(chess960_opening_boards)),
      # Divide by two as we consider both sides per opening (white and black).
      size=_NUM_GAMES.value // 2,
      replace=False,
  )

  chess960_opening_boards = list(chess960_opening_boards[idx] for idx in opening_indices)

  horde_opening_boards = [chess.variant.HordeBoard("rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1")] * (_NUM_GAMES.value // 2)


  if _VARIANT.value == 'standard':
      opening_boards = opening_boards
  elif _VARIANT.value == 'chess960':
      opening_boards = chess960_opening_boards
  elif _VARIANT.value == 'horde':
      opening_boards = horde_opening_boards
  else:
      raise ValueError(f'Unknown variant {_VARIANT.value}')
  
  if _VARIANT.value == 'standard' or _VARIANT.value == 'chess960':
      engines = {
        agent: constants.ENGINE_BUILDERS[agent]()
        for agent in [
            'BC_270M',
            'stockfish_1',
            'stockfish_2',
            'stockfish_3',
            'stockfish_4',
            'stockfish_5'
        ]
      }
  elif _VARIANT.value == 'horde':
      engines = {
        agent: constants.ENGINE_BUILDERS[agent]()
        for agent in [
            'BC_270M',
            'fairy_stockfish_1',
            'fairy_stockfish_2',
            'fairy_stockfish_3',
            'fairy_stockfish_4',
            'fairy_stockfish_5',
        ]
      }
  else:
      raise ValueError(f'Unknown variant {_VARIANT.value}')
  
  games = _run_tournament(engines=engines, opening_boards=opening_boards)

  games_path = os.path.join(os.getcwd(), f'{_VARIANT.value}_tournament_games.pgn')

  global count_move
  global count_ood
  global count_illegal
  print(count_ood)
  print(count_move)
  print(count_illegal)

  print(f'Writing games to {games_path}')
  with open(games_path, 'w') as file:
    for game in games:
      file.write(str(game))
      file.write('\n\n')


if __name__ == '__main__':
  app.run(main)
