# Copyright 2025 DeepMind Technologies Limited AND Anna Meszaros
#
# The original file was edited by A. Meszaros to support the evaluation of Stockfish with different
# multipv values (for top3, top5, top10 moves),skll levels, and ELO settings.



"""Implements a stockfish engine."""

import os

import chess
import chess.engine

from searchless_chess.src.engines import engine


class StockfishEngine(engine.Engine):
  """The classical version of stockfish."""

  def __init__(
      self,
      limit: chess.engine.Limit,
      skill_level: int | None = None,
      multipv: int = 1,
      elo: int | None = None
  ) -> None:
    self._limit = limit
    self._skill_level = skill_level
    self._multipv = multipv
    self._elo = elo
    bin_path = os.path.join(
        os.getcwd(),
        'Stockfish/src/stockfish',
    )
    self._raw_engine = chess.engine.SimpleEngine.popen_uci(bin_path)

    # Configure ELO if specified
    if self._elo is not None:
      self._raw_engine.configure({'UCI_Elo': self._elo})

  def __del__(self) -> None:
    self._raw_engine.close()

  @property
  def limit(self) -> chess.engine.Limit:
    return self._limit

  @property
  def skill_level(self) -> int | None:
    return self._skill_level

  @property
  def multipv(self) -> int:
    return self._multipv

  @skill_level.setter
  def skill_level(self, skill_level: int) -> None:
    self._skill_level = skill_level
    self._raw_engine.configure({'Skill Level': self._skill_level})

  def analyse(self, board: chess.Board) -> engine.AnalysisResult:
    """Returns analysis results from stockfish."""
    if self._skill_level is not None:
      self._raw_engine.configure({'Skill Level': self._skill_level})
    if self._multipv == 1:
      return self._raw_engine.analyse(board, limit=self._limit)
    else:
      return self._raw_engine.analyse(board, limit=self._limit, multipv=self._multipv)

  def play(self, board: chess.Board) -> chess.Move:
    """Returns the best move from stockfish."""
    if self._skill_level is not None:
      self._raw_engine.configure({'Skill Level': self._skill_level})
    best_move = self._raw_engine.play(board, limit=self._limit).move
    if best_move is None:
      raise ValueError('No best move found, something went wrong.')
    return best_move


class AllMovesStockfishEngine(StockfishEngine):
  """A version of stockfish that evaluates all moves individually."""

  def analyse(self, board: chess.Board) -> engine.AnalysisResult:
    """Returns analysis results from stockfish."""
    scores = []
    sorted_legal_moves = engine.get_ordered_legal_moves(board)
    for move in sorted_legal_moves:
      results = self._raw_engine.analyse(
          board,
          limit=self._limit,
          root_moves=[move],
      )
      scores.append((move, results['score'].relative))
    return {'scores': scores}

  def play(self, board: chess.Board) -> chess.Move:
    """Returns the best move from stockfish."""
    scores = self.analyse(board)['scores']
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[0][0]
