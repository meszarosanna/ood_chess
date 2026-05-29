"""Linear probing of BC_270M transformer hidden states for best-move squares.

For each transformer layer (0=embedding, 1-16=after each block), trains a
logistic regression binary classifier to predict whether each board square is
the from-square (or to-square) of the correct puzzle solution.

This is a task-specific probe: knowing the best move requires evaluating move
quality across the position, not just reading board structure. If this accuracy
peaks in later layers while attack-map accuracy peaks in early layers, it
provides a double dissociation supporting the hypothesis that:
  - Early layers: board structure (who attacks what)
  - Later layers: task-specific reasoning (which move is best)

Metric: balanced accuracy (mean of recall for class 0 and class 1), which
handles the severe class imbalance (only 1 positive square out of 64).

Run from /home/am3049/ood_chess:
    python rebuttal/probing_best_move.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import itertools
import chess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jax import random as jrandom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from searchless_chess.src import transformer
from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import utils
from searchless_chess.src.bagz import BagFileReader
from searchless_chess.src.constants import CODERS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'searchless_chess', 'checkpoints',
    'local', 'behavioral_cloning'
)
CHECKPOINT_STEP = 10_000_000
PUZZLES_CSV = os.path.join(
    os.path.dirname(__file__), '..', 'datasets', 'id_puzzles.csv'
)
OUTPUT_PDF = os.path.join(os.path.dirname(__file__), 'probing_best_move.pdf')


def _uci_square_to_tok(uci_sq: str) -> int:
    """Converts a UCI square string (e.g. 'e2') to tokenizer square index.

    Tokenizer order: a8=0, b8=1, ..., h8=7, a7=8, ..., h1=63.
    """
    file_ = ord(uci_sq[0]) - ord('a')   # 0=a .. 7=h
    rank  = int(uci_sq[1]) - 1          # 0=rank1 .. 7=rank8
    return (7 - rank) * 8 + file_


def build_bc_270m_predictor_with_hidden_states():
    predictor_config = transformer.TransformerConfig(
        vocab_size=len(tokenizer._CHARACTERS),
        output_size=utils.NUM_ACTIONS,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 1,
        num_heads=8,
        num_layers=16,
        embedding_dim=1024,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )
    predictor = transformer.build_transformer_predictor_with_hidden_states(
        config=predictor_config
    )
    dummy_params = predictor.initial_params(
        rng=jrandom.PRNGKey(1),
        targets=np.ones((1, 1), dtype=np.uint32),
    )
    params = training_utils.load_parameters(
        checkpoint_dir=CHECKPOINT_DIR,
        params=dummy_params,
        step=CHECKPOINT_STEP,
    )
    return predictor, params


def extract_hidden_states(predictor, params, fen_strings):
    """Returns all_hidden: shape (N, num_layers+1, 64, emb_dim)."""
    rng = jrandom.PRNGKey(0)
    all_hidden = []
    print(f'Extracting hidden states for {len(fen_strings)} positions...')
    for i, fen in enumerate(fen_strings):
        if i % 100 == 0:
            print(f'  {i}/{len(fen_strings)}')
        tok = tokenizer.tokenize(fen).astype(np.int32)
        targets = np.concatenate([tok, [0]], axis=0)[np.newaxis, :]
        _, hidden_states = predictor.predict(params, rng, targets)
        # Board squares at input positions 2..65 after shift_right.
        all_hidden.append(np.array(hidden_states)[:, 0, 2:66, :])
    all_hidden = np.stack(all_hidden, axis=0)  # (N, num_layers+1, 64, emb_dim)
    print('Done.')
    return all_hidden


def compute_best_move_labels(predictor, params, fen_strings):
    """Computes binary from/to labels using the transformer's top-1 predicted move.

    For each position the transformer's logits are used to select the best
    legal move, and that move's from- and to-squares are marked as positive.

    Returns:
        from_labels: np.ndarray, shape (N, 64), 1 at the from-square only
        to_labels:   np.ndarray, shape (N, 64), 1 at the to-square only
    """
    rng = jrandom.PRNGKey(0)
    N = len(fen_strings)
    from_labels = np.zeros((N, 64), dtype=np.int32)
    to_labels   = np.zeros((N, 64), dtype=np.int32)

    print(f'Computing transformer move predictions for {N} positions...')
    for i, fen in enumerate(fen_strings):
        if i % 100 == 0:
            print(f'  {i}/{N}')

        tok = tokenizer.tokenize(fen).astype(np.int32)
        targets = np.concatenate([tok, [0]], axis=0)[np.newaxis, :]
        logits, _ = predictor.predict(params, rng, targets)
        action_logits = np.array(logits[0, -1])  # (NUM_ACTIONS,)

        best_action = int(np.argmax(action_logits))
        best_move = chess.Move.from_uci(utils.ACTION_TO_MOVE[best_action])

        from_labels[i, _uci_square_to_tok(best_move.uci()[0:2])] = 1
        to_labels[i,   _uci_square_to_tok(best_move.uci()[2:4])] = 1

    print('Done.')
    return from_labels, to_labels


def train_probes(all_hidden, all_labels, label_name):
    """Trains one binary logistic regression probe per layer.

    Uses balanced accuracy as the metric (handles 1-vs-63 class imbalance).

    Returns:
        bal_accs: list of float, length num_layers+1
        random_bal_acc: float — balanced accuracy with shuffled labels
        majority_bal_acc: float — balanced accuracy of always predicting 0
    """
    N, num_layers_plus_one, _, emb_dim = all_hidden.shape

    idx = np.arange(N)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    y_test_flat  = all_labels[test_idx].reshape(-1)
    y_train_flat = all_labels[train_idx].reshape(-1)

    # Majority-class baseline: always predict 0 (no move).
    majority_pred = np.zeros_like(y_test_flat)
    majority_bal_acc = float(balanced_accuracy_score(y_test_flat, majority_pred))
    pos_frac = float(y_test_flat.mean())
    print(f'\n[{label_name}] Positive fraction: {pos_frac:.4f} '
          f'({int(y_test_flat.sum())} / {len(y_test_flat)} squares)')
    print(f'[{label_name}] Baseline — majority (always 0): '
          f'balanced acc = {majority_bal_acc:.4f}')

    # Random-label baseline (middle layer).
    mid_layer = num_layers_plus_one // 2
    X_train_mid_raw = all_hidden[train_idx, mid_layer].reshape(-1, emb_dim)
    X_test_mid_raw  = all_hidden[test_idx,  mid_layer].reshape(-1, emb_dim)
    scaler_mid = StandardScaler()
    X_train_mid = scaler_mid.fit_transform(X_train_mid_raw)
    X_test_mid  = scaler_mid.transform(X_test_mid_raw)
    rng_state = np.random.RandomState(0)
    clf_rand = LogisticRegression(
        max_iter=2000, C=1.0, solver='lbfgs',
        class_weight='balanced', n_jobs=-1
    )
    clf_rand.fit(X_train_mid, rng_state.permutation(y_train_flat))
    random_bal_acc = float(balanced_accuracy_score(
        y_test_flat, clf_rand.predict(X_test_mid)
    ))
    print(f'[{label_name}] Baseline — random labels (layer {mid_layer}): '
          f'balanced acc = {random_bal_acc:.4f}')

    bal_accs = []
    print(f'[{label_name}] Training {num_layers_plus_one} linear probes...')
    for layer_idx in range(num_layers_plus_one):
        X_train_raw = all_hidden[train_idx, layer_idx].reshape(-1, emb_dim)
        X_test_raw  = all_hidden[test_idx,  layer_idx].reshape(-1, emb_dim)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test  = scaler.transform(X_test_raw)

        # class_weight='balanced' upweights the rare positive class.
        clf = LogisticRegression(
            max_iter=2000, C=1.0, solver='lbfgs',
            class_weight='balanced', n_jobs=-1
        )
        clf.fit(X_train, y_train_flat)
        bal_acc = float(balanced_accuracy_score(y_test_flat, clf.predict(X_test)))
        bal_accs.append(bal_acc)

        tag = 'embed' if layer_idx == 0 else f'L{layer_idx}'
        print(f'  Layer {tag}: balanced acc = {bal_acc:.4f}')

    return bal_accs, majority_bal_acc, random_bal_acc


def plot_results(from_accs, to_accs, majority_bal_acc, random_bal_acc, output_path):
    """Saves the best-move probe balanced accuracy plot."""
    x = list(range(len(from_accs)))
    x_labels = ['emb'] + [str(i) for i in range(1, len(from_accs))]

    _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, from_accs, marker='o', color='steelblue', linewidth=2,
            markersize=5, label='From-square probe')
    ax.plot(x, to_accs, marker='s', color='coral', linewidth=2,
            markersize=5, label='To-square probe')
    ax.axhline(y=majority_bal_acc, color='darkorange', linestyle='--',
               linewidth=1.2, label=f'Majority class ({majority_bal_acc:.2f})')
    ax.axhline(y=random_bal_acc, color='gray', linestyle=':',
               linewidth=1.2, label=f'Random labels ({random_bal_acc:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel('Layer (0 = embedding, 1–16 = transformer blocks)')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('Linear probe: transformer predicted move from/to square\n(BC-270M, id_puzzles)')
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f'\nSaved figure to {output_path}')


def main():
    print('Building predictor with hidden states...')
    predictor, params = build_bc_270m_predictor_with_hidden_states()
    print('Model loaded.')

    # Load id_puzzles, pushing the first move to get the actual puzzle position.
    import pandas as pd
    df = pd.read_csv(PUZZLES_CSV, nrows=1000)
    fen_strings = []
    for _, row in df.iterrows():
        board = chess.Board(row['FEN'])
        board.push(chess.Move.from_uci(row['Moves'].split()[0]))
        fen_strings.append(board.fen())
    print(f'Loaded {len(fen_strings)} positions from {PUZZLES_CSV}')

    all_hidden = extract_hidden_states(predictor, params, fen_strings)
    print(f'Hidden states shape: {all_hidden.shape}')

    # Build labels directly from the decoded UCI moves (no split needed).
    print('\nComputing best-move labels (transformer predictions)...')
    from_labels, to_labels = compute_best_move_labels(predictor, params, fen_strings)

    from_accs, majority_bal_acc, random_bal_acc = train_probes(
        all_hidden, from_labels, label_name='from-square'
    )
    to_accs, _, _ = train_probes(
        all_hidden, to_labels, label_name='to-square'
    )

    plot_results(from_accs, to_accs, majority_bal_acc, random_bal_acc, OUTPUT_PDF)

    print('\n=== Summary (balanced accuracy) ===')
    print(f'{"Layer":<8} {"From-sq":>10} {"To-sq":>10}')
    for i, (fa, ta) in enumerate(zip(from_accs, to_accs)):
        tag = 'embed' if i == 0 else f'layer {i:2d}'
        print(f'{tag:<8} {fa:>10.4f} {ta:>10.4f}')
    print(f'{"majority":<8} {majority_bal_acc:>10.4f} {majority_bal_acc:>10.4f}')
    print(f'{"rand-lbl":<8} {random_bal_acc:>10.4f} {random_bal_acc:>10.4f}')


if __name__ == '__main__':
    main()
