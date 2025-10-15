# Copyright 2025 DeepMind Technologies Limited AND Meszaros et al.
#
# The original file was edited by Anna Meszaros to support the BC_270M model (trained on the filtered dataset) and
# the tournaments with stockfishes and fairy-stockfishes
#


"""Constants for the engines."""

import functools
import os

import chess
import chess.engine
import chess.pgn
from jax import random as jrandom
import numpy as np

from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import lc0_engine
from searchless_chess.src.engines import neural_engines
from searchless_chess.src.engines import stockfish_engine
from searchless_chess.src.engines import fairy_stockfish_engine

def _build_neural_engine(
    model_name: str,
    checkpoint_step: int = -1,
) -> neural_engines.NeuralEngine:
  """Returns a neural engine."""

  match model_name:
    case '9M':
      policy = 'action_value'
      num_layers = 8
      embedding_dim = 256
      num_heads = 8
    case '9M BC':
      policy = 'behavioral_cloning'
      num_layers = 8
      embedding_dim = 256
      num_heads = 8
    case '136M':
      policy = 'action_value'
      num_layers = 8
      embedding_dim = 1024
      num_heads = 8
    case '270M':
      policy = 'action_value'
      num_layers = 16
      embedding_dim = 1024
      num_heads = 8
    case 'local':
      policy = 'action_value'
      num_layers = 4
      embedding_dim = 64
      num_heads = 4
    case 'BC_270M':
      policy = 'behavioral_cloning'
      num_layers = 16
      embedding_dim = 1024
      num_heads = 8
    case _:
      raise ValueError(f'Unknown model: {model_name}')

  num_return_buckets = 128

  match policy:
    case 'action_value':
      output_size = num_return_buckets
      vocab_size = utils.NUM_ACTIONS
      max_sequence_length = tokenizer.SEQUENCE_LENGTH + 2
    case 'behavioral_cloning':
      output_size = utils.NUM_ACTIONS 
      vocab_size = len(tokenizer._CHARACTERS)
      max_sequence_length = tokenizer.SEQUENCE_LENGTH + 1
    case 'state_value':
      output_size = num_return_buckets
      vocab_size = len(tokenizer._CHARACTERS)
      max_sequence_length = tokenizer.SEQUENCE_LENGTH + 1

  predictor_config = transformer.TransformerConfig(
      vocab_size=vocab_size,
      output_size=output_size,
      pos_encodings=transformer.PositionalEncodings.LEARNED,
      max_sequence_length=max_sequence_length,
      num_heads=num_heads,
      num_layers=num_layers,
      embedding_dim=embedding_dim,
      apply_post_ln=True,
      apply_qk_layernorm=False,
      use_causal_mask=False,
  )

  predictor = transformer.build_transformer_predictor(config=predictor_config)
  if model_name == 'BC_270M':
    checkpoint_dir = os.path.join(
        os.getcwd(),
        'searchless_chess/checkpoints/local/behavioral_cloning',
    )
  else:
    checkpoint_dir = os.path.join(
    os.getcwd(),
    f'searchless_chess/checkpoints/{model_name}',
    )
    
  params = training_utils.load_parameters(
      checkpoint_dir=checkpoint_dir,
      params=predictor.initial_params(
          rng=jrandom.PRNGKey(1),
          targets=np.ones((1, 1), dtype=np.uint32),
      ),
      step=checkpoint_step,
  )
  _, return_buckets_values = utils.get_uniform_buckets_edges_values(
      num_return_buckets
  )
  return neural_engines.ENGINE_FROM_POLICY[policy](
      return_buckets_values=return_buckets_values,
      predict_fn=neural_engines.wrap_predict_fn(
          predictor=predictor,
          params=params,
          batch_size=1,
      ),
  )


ENGINE_BUILDERS = {
    'local': functools.partial(_build_neural_engine, model_name='local'),
    '9M': functools.partial(
        _build_neural_engine, model_name='9M', checkpoint_step=6_400_000
    ),
    '136M': functools.partial(
        _build_neural_engine, model_name='136M', checkpoint_step=6_400_000
    ),
    '270M': functools.partial(
        _build_neural_engine, model_name='270M', checkpoint_step=6_400_000
    ),
    'BC_270M': functools.partial(
        _build_neural_engine, model_name='BC_270M', checkpoint_step=10_000_000
    ),
    'stockfish': lambda: stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(depth=30)         
    ),
    'stockfish_sort': lambda: stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(time=0.05)
    ),
    'stockfish_top3': lambda: stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(depth=20), multipv=3
    ),
    'stockfish_top5': lambda: stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(depth=20), multipv=5
    ),
    'stockfish_top10': lambda: stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(depth=20), multipv=10
    ),
    'stockfish_all_moves': lambda: stockfish_engine.AllMovesStockfishEngine(
        limit=chess.engine.Limit(time=0.05)
    ),
    'leela_chess_zero_depth_1': lambda: lc0_engine.AllMovesLc0Engine(
        limit=chess.engine.Limit(nodes=1),
    ),
    'leela_chess_zero_policy_net': lambda: lc0_engine.Lc0Engine(
        limit=chess.engine.Limit(nodes=1),
    ),
    'leela_chess_zero_400_sims': lambda: lc0_engine.Lc0Engine(
        limit=chess.engine.Limit(nodes=400),
    ),
    'stockfish_1': lambda: stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(time=0.05), skill_level=0
    ),
    'stockfish_2': lambda: stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(time=0.05), skill_level=1
    ),
    'stockfish_3': lambda: stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(time=0.05), skill_level=2
    ),
    'stockfish_4': lambda: stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(time=0.05), skill_level=3
    ),
    'stockfish_5': lambda: stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(time=0.05), skill_level=4
    ),
    'fairy_stockfish_1': lambda: fairy_stockfish_engine.FairyStockfishEngine(
        limit=chess.engine.Limit(time=0.05), skill_level=0, variant = "horde"
    ),
    'fairy_stockfish_2': lambda: fairy_stockfish_engine.FairyStockfishEngine(
        limit=chess.engine.Limit(time=0.05), skill_level=1, variant = "horde"
    ),
    'fairy_stockfish_3': lambda: fairy_stockfish_engine.FairyStockfishEngine(
        limit=chess.engine.Limit(time=0.05), skill_level=2, variant = "horde"
    ),
    'fairy_stockfish_4': lambda: fairy_stockfish_engine.FairyStockfishEngine(
        limit=chess.engine.Limit(time=0.05), skill_level=3, variant = "horde"
    ),
    'fairy_stockfish_5': lambda: fairy_stockfish_engine.FairyStockfishEngine(
        limit=chess.engine.Limit(time=0.05), skill_level=4, variant = "horde"
    )
}
