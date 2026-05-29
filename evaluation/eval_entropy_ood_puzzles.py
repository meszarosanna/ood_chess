# Copyright 2025 Anna Meszaros

"""Compute the mean entropy of the transformer's output probability distribution
across multiple datasets."""

from collections.abc import Sequence
import chess
import pandas as pd
from tqdm import tqdm
from absl import app
import itertools
import math
import numpy as np
from scipy.stats import entropy
from searchless_chess.src.engines import constants
from searchless_chess.src.bagz import BagFileReader
from searchless_chess.src.constants import CODERS

MODEL_NAME = 'BC_270M'
neural_engine = constants.ENGINE_BUILDERS[MODEL_NAME]()

_NUM_PUZZLES = 1000

DATASETS = [
    'id_puzzles.csv',
    'filtered_behavioral_cloning_test_data.bag',
    'ood_puzzles.csv',
    'more_pieces_puzzles.csv',
    'same_color_puzzles.csv',
    'chess960_puzzles.csv',
    'all_ordering_puzzles.csv',
    'knights_and_rooks.csv',
]


def compute_entropy(board: chess.Board) -> float:
    log_probs = neural_engine.analyse(board, legal=False)
    probs = np.exp(np.array(log_probs))
    return entropy(probs, base=math.e)


def mean_entropy_for_dataset(name: str) -> float:
    path = 'datasets/' + name
    entropies = []

    if name.endswith('.bag'):
        reader = BagFileReader(path)
        for record in tqdm(itertools.islice(reader, _NUM_PUZZLES), total=_NUM_PUZZLES, desc=name):
            fen, _ = CODERS['behavioral_cloning'].decode(record)
            board = chess.Board(fen)
            entropies.append(compute_entropy(board))

    elif name in ('id_puzzles.csv', 'ood_puzzles.csv'):
        puzzles = pd.read_csv(path, nrows=_NUM_PUZZLES)
        for _, puzzle in tqdm(puzzles.iterrows(), total=len(puzzles), desc=name):
            board = chess.Board(puzzle['FEN'])
            board.push(chess.Move.from_uci(puzzle['Moves'].split(' ')[0]))
            entropies.append(compute_entropy(board))

    else:
        puzzles = pd.read_csv(path, nrows=_NUM_PUZZLES)
        for _, puzzle in tqdm(puzzles.iterrows(), total=len(puzzles), desc=name):
            board = chess.Board(puzzle['FEN'])
            entropies.append(compute_entropy(board))

    return float(np.mean(entropies))


def main(argv: Sequence[str]) -> None:
    print(f"Computing mean entropy over {_NUM_PUZZLES} samples per dataset\n")
    results = {}
    for name in DATASETS:
        results[name] = mean_entropy_for_dataset(name)

    print(f"\n{'Dataset':<45} {'Mean Entropy (nats)':>20}")
    print('-' * 67)
    for name, ent in results.items():
        print(f"{name:<45} {ent:>20.6f}")


if __name__ == '__main__':
    app.run(main)
