# Copyright 2025 DeepMind Technologies Limited AND Anna Meszaros
#
# The original file was edited by A. Meszaros to support the training of the BC_270M model on the filtered dataset
#


"""An example training script."""

from collections.abc import Sequence

from absl import app
from absl import flags

from searchless_chess.src import config as config_lib
from searchless_chess.src import data_loader
from searchless_chess.src import metrics_evaluator
from searchless_chess.src import tokenizer
from searchless_chess.src import training
from searchless_chess.src import transformer
from searchless_chess.src import utils

_POLICY = flags.DEFINE_enum(
    'policy',
    'behavioral_cloning',
    config_lib.POLICY_TYPES,
    'The policy used to play moves with the model.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  policy: config_lib.PolicyType = _POLICY.value  # pytype: disable=annotation-type-mismatch
  num_return_buckets = 128

  match policy:
    case 'action_value':
      vocab_size = utils.NUM_ACTIONS
      output_size = num_return_buckets
      max_sequence_length = tokenizer.SEQUENCE_LENGTH + 2
    case 'behavioral_cloning':
      vocab_size = len(tokenizer._CHARACTERS)
      output_size = utils.NUM_ACTIONS
      max_sequence_length = tokenizer.SEQUENCE_LENGTH + 1
    case 'state_value':
      vocab_size = utils.NUM_ACTIONS
      output_size = num_return_buckets
      max_sequence_length = tokenizer.SEQUENCE_LENGTH + 1

  predictor_config = transformer.TransformerConfig(
      vocab_size=vocab_size,
      output_size=output_size,
      pos_encodings=transformer.PositionalEncodings.LEARNED,
      max_sequence_length=max_sequence_length,
      num_heads=8,
      num_layers=16,
      embedding_dim=1024,
      apply_post_ln=True,
      apply_qk_layernorm=False,
      use_causal_mask=False,
  )
  train_config = config_lib.TrainConfig(
      learning_rate=1e-4,
      data=config_lib.DataConfig(
          batch_size=256, #256
          shuffle=True,
          worker_count=0,  # 0 disables multiprocessing.
          num_return_buckets=num_return_buckets,
          policy=policy,
          split='train',
      ),
      log_frequency=1000,
      num_steps=10000000,
      ckpt_frequency=1000000,
      save_frequency=1000000, 
  )
  eval_config = config_lib.EvalConfig(
      data=config_lib.DataConfig(
          batch_size=1,
          shuffle=False,
          worker_count=0,  # 0 disables multiprocessing.
          num_return_buckets=num_return_buckets,
          policy=None,  # pytype: disable=wrong-arg-types
          split='test',
      ),
      use_ema_params=True,
      policy=policy,
      batch_size=256,
      num_return_buckets=num_return_buckets,
      num_eval_data=64,
  )

  params = training.train(
      train_config=train_config,
      predictor_config=predictor_config,
      build_data_loader=data_loader.build_data_loader,
  )

  predictor = transformer.build_transformer_predictor(predictor_config)
  evaluator = metrics_evaluator.build_evaluator(predictor, eval_config)
  print(evaluator.step(params=params, step=train_config.num_steps))


if __name__ == '__main__':
  app.run(main)
