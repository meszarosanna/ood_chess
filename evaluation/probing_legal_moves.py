"""Linear probing of BC_270M transformer hidden states for legal move counts.

For each transformer layer (0=embedding, 1-16=after each block), trains a
logistic regression classifier to predict the number of legal moves available
from each board square, using the residual-stream hidden state at that square's
token position as input.

Legal move count requires reasoning about piece type, all blocking/pinning
pieces, and check constraints — it cannot be read from a single token.
High accuracy above baseline is mechanistic evidence that the model computes
chess structure in its hidden states.

Label: number of legal moves originating from that square (0 for empty squares,
opponent pieces, or fully pinned pieces). Binned into 5 buckets:
    0 = 0 moves  (empty or no legal moves)
    1 = 1–2 moves
    2 = 3–5 moves
    3 = 6–10 moves
    4 = 11+ moves

Run from /home/am3049/ood_chess:
    python rebuttal/probing_legal_moves.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import chess
import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'searchless_chess', 'checkpoints',
    'local', 'behavioral_cloning'
)
CHECKPOINT_STEP = 10_000_000
PUZZLES_CSV = os.path.join(
    os.path.dirname(__file__), '..', 'datasets', 'ood_puzzles.csv'
)
OUTPUT_PDF = os.path.join(os.path.dirname(__file__), 'probing_legal_moves.pdf')

# Bin edges for legal move counts (right-exclusive).
# 0 moves | 1-2 | 3-5 | 6-10 | 11+
_BINS = [0, 1, 3, 6, 11]   # bin i: _BINS[i] <= count < _BINS[i+1] (last bin: 11+)
_BIN_LABELS = ['0', '1–2', '3–5', '6–10', '11+']


def _count_to_bin(count: int) -> int:
    for i in range(len(_BINS) - 1, -1, -1):
        if count >= _BINS[i]:
            return i
    return 0


def build_bc_270m_predictor_with_hidden_states():
    """Builds the BC_270M predictor that returns (logits, hidden_states)."""
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
    """Runs forward passes and collects hidden states for all positions.

    Returns:
        all_hidden: np.ndarray, shape (N, num_layers+1, 64, embedding_dim)
            Hidden states at the 64 board-square token positions (input pos 2-65).
    """
    rng = jrandom.PRNGKey(0)
    all_hidden = []

    print(f'Extracting hidden states for {len(fen_strings)} positions...')
    for i, fen in enumerate(fen_strings):
        if i % 100 == 0:
            print(f'  {i}/{len(fen_strings)}')

        tok = tokenizer.tokenize(fen).astype(np.int32)  # (77,)
        targets = np.concatenate([tok, [0]], axis=0)[np.newaxis, :]  # (1, 78)

        _logits, hidden_states = predictor.predict(params, rng, targets)
        # After shift_right: board square s (0-indexed, 0=a8) is at input pos s+2.
        all_hidden.append(np.array(hidden_states)[:, 0, 2:66, :])

    all_hidden = np.stack(all_hidden, axis=0)  # (N, num_layers+1, 64, emb_dim)
    print('Done.')
    return all_hidden


def compute_legal_move_labels(fen_strings):
    """Computes binned legal-move-count labels for each board square.

    For the side to move, counts how many legal moves originate from each
    square, then bins the count (see _BINS / _BIN_LABELS).

    Returns:
        labels: np.ndarray, shape (N, 64), dtype int32
            Bin index (0–4) for each square.
    """
    N = len(fen_strings)
    labels = np.zeros((N, 64), dtype=np.int32)
    for i, fen in enumerate(fen_strings):
        board = chess.Board(fen)
        counts = np.zeros(64, dtype=np.int32)  # indexed in tokenizer order
        for move in board.legal_moves:
            # python-chess square: a1=0 .. h8=63
            # Convert to tokenizer order (a8=0 .. h1=63)
            sq = move.from_square
            rank = sq // 8   # 0=rank1, 7=rank8
            file_ = sq % 8
            tok_sq = (7 - rank) * 8 + file_
            counts[tok_sq] += 1
        labels[i] = np.array([_count_to_bin(c) for c in counts])
    return labels


# Piece-type categories occupying a square, color-aware (P and p are distinct,
# matching the input token): 0 = empty, 1-6 = white P/N/B/R/Q/K,
# 7-12 = black p/n/b/r/q/k. 13 categories total.
_NUM_PIECE_CATS = 13


def compute_piece_categories(fen_strings):
    """Piece-type category occupying each square, in tokenizer order.

    Color-aware: white and black of the same piece type are distinct categories,
    matching the per-square input token (which encodes piece color).

    Returns:
        cats: np.ndarray, shape (N, 64), dtype int32 — category in [0, 12].
    """
    N = len(fen_strings)
    cats = np.zeros((N, 64), dtype=np.int32)
    for i, fen in enumerate(fen_strings):
        board = chess.Board(fen)
        for s in range(64):
            rank = 7 - (s // 8)
            file_ = s % 8
            piece = board.piece_at(chess.square(file_, rank))
            if piece is not None:
                base = 0 if piece.color == chess.WHITE else 6
                cats[i, s] = base + piece.piece_type  # piece_type in 1..6
    return cats


def _balanced_group_prediction(group_train, y_train, group_test,
                               num_groups, num_classes):
    """Strongest trivial baseline using only a categorical group feature.

    Mirrors the probe's class_weight='balanced': for each group, predicts the
    class maximizing the inverse-class-frequency-weighted count, rather than the
    plain mode. This is the balanced-accuracy analog of a per-group majority and
    equivalent to a class-weighted classifier whose only input is the one-hot
    group identity. Rare classes are upweighted so they can actually be
    predicted, giving a meaningful score under balanced accuracy.

    Returns:
        np.ndarray of predicted classes for group_test.
    """
    class_counts = np.bincount(y_train, minlength=num_classes)
    weights = len(y_train) / (num_classes * np.maximum(class_counts, 1))
    pred_by_group = np.zeros(num_groups, dtype=np.int64)
    for g in range(num_groups):
        mask = group_train == g
        if mask.any():
            counts = np.bincount(y_train[mask], minlength=num_classes)
            pred_by_group[g] = int(np.argmax(counts * weights))
    return pred_by_group[group_test]


def train_probes(all_hidden, all_labels, piece_cats):
    """Trains one logistic regression probe per layer.

    Args:
        all_hidden: shape (N, num_layers+1, 64, emb_dim)
        all_labels: shape (N, 64) — integer bin labels
        piece_cats: shape (N, 64) — piece-type category per square (see
            compute_piece_categories)

    Returns:
        accuracies: list of float, length num_layers+1
        majority_acc: float — accuracy of always predicting the most common bin
        random_label_acc: float — probe accuracy trained on shuffled labels
        per_square_acc: float — balanced accuracy of the per-square positional
            prior (predict each square's most frequent train-set bin)
        per_piece_acc: float — balanced accuracy of the per-piece-type prior
            (predict each piece type's most frequent train-set bin)
    """
    N, num_layers_plus_one, _, emb_dim = all_hidden.shape

    idx = np.arange(N)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    y_test_flat = all_labels[test_idx].reshape(-1)
    y_train_flat = all_labels[train_idx].reshape(-1)

    num_classes = len(_BIN_LABELS)

    # Majority-class baseline: always predict the most common bin.
    values, counts = np.unique(y_test_flat, return_counts=True)
    majority_class = values[np.argmax(counts)]
    majority_pred = np.full_like(y_test_flat, majority_class)
    majority_acc = float(balanced_accuracy_score(y_test_flat, majority_pred))
    print(f'\nBaseline — majority class (bin {majority_class} = '
          f'"{_BIN_LABELS[majority_class]}" moves): balanced acc = {majority_acc:.4f}')

    # Random-label baseline (middle layer).
    mid_layer = num_layers_plus_one // 2
    X_train_mid_raw = all_hidden[train_idx, mid_layer].reshape(-1, emb_dim)
    X_test_mid_raw = all_hidden[test_idx, mid_layer].reshape(-1, emb_dim)
    scaler_mid = StandardScaler()
    X_train_mid = scaler_mid.fit_transform(X_train_mid_raw)
    X_test_mid = scaler_mid.transform(X_test_mid_raw)
    rng_state = np.random.RandomState(0)
    clf_rand = LogisticRegression(
        max_iter=5000, C=1.0, solver='saga',
        class_weight='balanced', n_jobs=-1
    )
    clf_rand.fit(X_train_mid, rng_state.permutation(y_train_flat))
    random_label_acc = float(balanced_accuracy_score(
        y_test_flat, clf_rand.predict(X_test_mid)
    ))
    print(f'Baseline — random labels (layer {mid_layer}): {random_label_acc:.4f}')

    # Per-square positional-prior baseline: predict, for each of the 64 squares,
    # the class-balanced most-frequent bin (inverse-frequency weighted, matching
    # the probe's class_weight='balanced'). Scored with balanced accuracy.
    sq_train = np.tile(np.arange(64), len(train_idx))
    sq_test = np.tile(np.arange(64), len(test_idx))
    per_sq_pred = _balanced_group_prediction(
        sq_train, y_train_flat, sq_test, 64, num_classes)
    per_square_acc = float(balanced_accuracy_score(y_test_flat, per_sq_pred))
    print(f'Baseline — per-square (balanced): balanced acc = {per_square_acc:.4f}')

    # Per-piece-type prior baseline: group squares by the piece occupying them
    # and predict each group's class-balanced most-frequent bin. Legal-move count
    # is strongly piece-dependent (empty=0, pawn few, queen many), so this is a
    # demanding baseline.
    cats_train = piece_cats[train_idx].reshape(-1)
    cats_test = piece_cats[test_idx].reshape(-1)
    per_piece_pred = _balanced_group_prediction(
        cats_train, y_train_flat, cats_test, _NUM_PIECE_CATS, num_classes)
    per_piece_acc = float(balanced_accuracy_score(y_test_flat, per_piece_pred))
    print(f'Baseline — per-piece-type (balanced): balanced acc = {per_piece_acc:.4f}')

    accuracies = []
    print(f'\nTraining {num_layers_plus_one} linear probes...')
    for layer_idx in range(num_layers_plus_one):
        X_train_raw = all_hidden[train_idx, layer_idx].reshape(-1, emb_dim)
        X_test_raw = all_hidden[test_idx, layer_idx].reshape(-1, emb_dim)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        clf = LogisticRegression(
            max_iter=5000, C=1.0, solver='saga',
            class_weight='balanced', n_jobs=-1
        )
        clf.fit(X_train, y_train_flat)
        acc = float(balanced_accuracy_score(y_test_flat, clf.predict(X_test)))
        accuracies.append(acc)

        tag = 'embed' if layer_idx == 0 else f'L{layer_idx}'
        print(f'  Layer {tag}: balanced acc = {acc:.4f}')

    return accuracies, majority_acc, random_label_acc, per_square_acc, per_piece_acc


def plot_results(accuracies, majority_acc, random_label_acc, per_square_acc,
                 per_piece_acc, output_path):
    """Saves the legal-move-count probe accuracy plot."""
    x = list(range(len(accuracies)))
    x_labels = ['emb'] + [str(i) for i in range(1, len(accuracies))]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, accuracies, marker='o', color='mediumpurple', linewidth=2,
            markersize=5, label='Probe accuracy')
    ax.axhline(y=per_square_acc, color='steelblue', linestyle='-.', linewidth=1.2,
               label=f'Per-square majority ({per_square_acc:.2f})')
    ax.axhline(y=per_piece_acc, color='crimson', linestyle=(0, (3, 1, 1, 1)),
               linewidth=1.2, label=f'Per-piece majority ({per_piece_acc:.2f})')
    ax.axhline(y=majority_acc, color='darkorange', linestyle='--', linewidth=1.2,
               label=f'Majority class ({majority_acc:.2f})')
    ax.axhline(y=random_label_acc, color='gray', linestyle=':', linewidth=1.2,
               label=f'Random labels ({random_label_acc:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel('Layer (0 = embedding, 1–16 = transformer blocks)')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('Linear probe: legal move count per square\n(BC-270M, id_puzzles)')
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

    # Puzzles in the CSV store the FEN *before* the opponent's setup move;
    # the actual puzzle position is obtained by pushing moves[0].
    df = pd.read_csv(PUZZLES_CSV, nrows=1000)
    fen_strings = []
    for _, row in df.iterrows():
        board = chess.Board(row['FEN'])
        board.push(chess.Move.from_uci(row['Moves'].split()[0]))
        fen_strings.append(board.fen())
    print(f'Loaded {len(fen_strings)} FEN positions from {PUZZLES_CSV}')

    all_hidden = extract_hidden_states(predictor, params, fen_strings)
    print(f'Hidden states shape: {all_hidden.shape}')

    print('\nComputing legal-move-count labels...')
    labels = compute_legal_move_labels(fen_strings)
    # Print bin distribution for reference.
    flat = labels.reshape(-1)
    for b, name in enumerate(_BIN_LABELS):
        print(f'  bin {b} ({name} moves): {(flat == b).sum()} squares '
              f'({100*(flat == b).mean():.1f}%)')

    # Piece-type category per square (for the per-piece-type baseline).
    piece_cats = compute_piece_categories(fen_strings)

    (accuracies, majority_acc, random_label_acc,
     per_square_acc, per_piece_acc) = train_probes(all_hidden, labels, piece_cats)

    plot_results(accuracies, majority_acc, random_label_acc, per_square_acc,
                 per_piece_acc, OUTPUT_PDF)

    print('\n=== Summary ===')
    print(f'{"Layer":<8} {"Accuracy":>10}')
    for i, acc in enumerate(accuracies):
        tag = 'embed' if i == 0 else f'layer {i:2d}'
        print(f'{tag:<8} {acc:>10.4f}')
    print(f'{"majority":<8} {majority_acc:>10.4f}')
    print(f'{"rand-lbl":<8} {random_label_acc:>10.4f}')
    print(f'{"per-sq":<8} {per_square_acc:>10.4f}')
    print(f'{"per-piece":<8} {per_piece_acc:>10.4f}')


if __name__ == '__main__':
    main()
