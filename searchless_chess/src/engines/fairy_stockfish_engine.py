# Copyright Anna Meszaros
#
# For simplicity, the file is in the searchless_chess folder, but it was created by A. Meszaros.
#

"""Implements a fairy stockfish engine for chess variants."""

import os
import chess
import chess.engine

from searchless_chess.src.engines import engine


class FairyStockfishEngine(engine.Engine):
  """The fairy stockfish engine for chess variants."""

  def __init__(
      self,
      limit: chess.engine.Limit,
      skill_level: int | None = None,
      multipv: int = 1,
      variant: str = "horde",
      elo: int | None = None
  ) -> None:
    self._limit = limit
    self._skill_level = skill_level
    self._multipv = multipv
    self._variant = variant
    self._elo = elo
    bin_path = os.path.join(
        os.getcwd(),
        'Fairy-Stockfish/src/stockfish',
    )
    
    # Create the engine using chess.engine
    self._raw_engine = chess.engine.SimpleEngine.popen_uci(bin_path)
    
    # Configure ELO if specified
    if self._elo is not None:
      self._raw_engine.configure({'UCI_Elo': self._elo})
    

  def __del__(self) -> None:
    if hasattr(self, '_raw_engine'):
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

  @property
  def variant(self) -> str:
    return self._variant

  @property
  def elo(self) -> int | None:
    return self._elo

  @skill_level.setter
  def skill_level(self, skill_level: int) -> None:
    self._skill_level = skill_level
    self._raw_engine.configure({'Skill Level': self._skill_level})

  @elo.setter
  def elo(self, elo: int) -> None:
    self._elo = elo
    self._raw_engine.configure({'UCI_Elo': self._elo})

  def analyse(self, board: chess.Board) -> engine.AnalysisResult:
    """Returns analysis results from fairy stockfish."""
    
    if self._skill_level is not None:
      self._raw_engine.configure({'Skill Level': self._skill_level})
    if self._multipv == 1:
      return self._raw_engine.analyse(board, limit=self._limit)
    else:
      return self._raw_engine.analyse(board, limit=self._limit, multipv=self._multipv)

  def play(self, board: chess.Board) -> chess.Move:
    """Returns the best move from fairy stockfish."""
    if self._skill_level is not None:
      self._raw_engine.configure({'Skill Level': self._skill_level})
    best_move = self._raw_engine.play(board, limit=self._limit).move
    if best_move is None:
      raise ValueError('No best move found, something went wrong.')
    return best_move


class AllMovesFairyStockfishEngine(FairyStockfishEngine):
  """A version of fairy stockfish that evaluates all moves individually."""

  def analyse(self, board: chess.Board) -> engine.AnalysisResult:
    """Returns analysis results from fairy stockfish."""
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
    """Returns the best move from fairy stockfish."""
    scores = self.analyse(board)['scores']
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[0][0]
