"""Linear probing of BC_270M transformer hidden states for attack maps.

For each transformer layer (0=embedding, 1-16=after each block), trains a
logistic regression classifier to predict which squares are attacked from the
residual-stream hidden state at each board square's token position.

Attack maps require board-wide reasoning (piece positions + movement rules),
so high accuracy provides genuine mechanistic evidence that the model computes
chess structure, not just memorizes token patterns.

Run from /home/am3049/ood_chess:
    python rebuttal/probing.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import chess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jax import random as jrandom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from searchless_chess.src import transformer
from searchless_chess.src import tokenizer
from searchless_chess.src import training_utils
from searchless_chess.src import utils
import pandas as pd


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
OUTPUT_PDF = os.path.join(os.path.dirname(__file__), 'probing_results.pdf')

# Attack label classes:
#   0 = not attacked by either side
#   1 = attacked by White only
#   2 = attacked by Black only
#   3 = attacked by both sides
_ATTACK_LABEL_NAMES = ['none', 'white only', 'black only', 'both']


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
    # Initialize dummy params to get the parameter structure for checkpoint loading.
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
        all_hidden: np.ndarray, shape [N, num_layers+1, 64, embedding_dim]
            Hidden states at the 64 board-square token positions (input pos 2-65).
    """
    rng = jrandom.PRNGKey(0)
    all_hidden = []

    print(f'Extracting hidden states for {len(fen_strings)} positions...')
    for i, fen in enumerate(fen_strings):
        if i % 100 == 0:
            print(f'  {i}/{len(fen_strings)}')

        tok = tokenizer.tokenize(fen).astype(np.int32)  # shape (77,)
        # Append a dummy action token (BC model expects seq len 78).
        targets = np.concatenate([tok, [0]], axis=0)[np.newaxis, :]  # (1, 78)

        _logits, hidden_states = predictor.predict(params, rng, targets)
        # hidden_states: (num_layers+1, 1, 78, 1024)
        # After shift_right, board square s (0-indexed, 0=a8) lives at input
        # position s+2. Extract positions 2..65: (num_layers+1, 64, emb_dim)
        all_hidden.append(np.array(hidden_states)[:, 0, 2:66, :])

    all_hidden = np.stack(all_hidden, axis=0)  # (N, num_layers+1, 64, emb_dim)
    print('Done.')
    return all_hidden


def compute_attack_labels(fen_strings):
    """Computes attack-map labels for each position.

    For each of the 64 board squares (in tokenizer order: a8=0 .. h1=63),
    assigns a 4-class label based on which side(s) attack that square:
        0 = not attacked by either side
        1 = attacked by White only
        2 = attacked by Black only
        3 = attacked by both sides

    Args:
        fen_strings: list of FEN strings, length N

    Returns:
        attack_labels: np.ndarray, shape (N, 64), dtype int32
    """
    N = len(fen_strings)
    attack_labels = np.zeros((N, 64), dtype=np.int32)
    for i, fen in enumerate(fen_strings):
        board = chess.Board(fen)
        for s in range(64):
            # Convert tokenizer square index to python-chess square index.
            # Tokenizer: s=0 is a8 (rank 8, file a), s=63 is h1 (rank 1, file h).
            # python-chess: sq=56 is a8, sq=7 is h1.
            rank = 7 - (s // 8)   # 0=rank1 .. 7=rank8
            file_ = s % 8          # 0=file_a .. 7=file_h
            sq = chess.square(file_, rank)
            by_white = board.is_attacked_by(chess.WHITE, sq)
            by_black = board.is_attacked_by(chess.BLACK, sq)
            if by_white and by_black:
                attack_labels[i, s] = 3
            elif by_white:
                attack_labels[i, s] = 1
            elif by_black:
                attack_labels[i, s] = 2
            # else: 0 (not attacked)
    return attack_labels


def train_probes(all_hidden, all_labels, probe_name='probe'):
    """Trains one logistic regression probe per layer.

    Args:
        all_hidden: shape (N, num_layers+1, 64, emb_dim)
        all_labels: shape (N, 64) — integer class labels
        probe_name: string used in print output

    Returns:
        accuracies: list of float, length num_layers+1
        majority_acc: float — accuracy of always predicting the most common class
        random_label_acc: float — probe accuracy when trained on shuffled labels
    """
    N, num_layers_plus_one, _, emb_dim = all_hidden.shape

    # Fixed position-level train/test split (same for all layers).
    idx = np.arange(N)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    y_test_flat = all_labels[test_idx].reshape(-1)
    y_train_flat = all_labels[train_idx].reshape(-1)

    # Majority-class baseline: always predict the most common class.
    values, counts = np.unique(y_test_flat, return_counts=True)
    majority_class = values[np.argmax(counts)]
    print(f'Most common class in test set: {majority_class} ({_ATTACK_LABEL_NAMES[majority_class]})')
    majority_acc = float((y_test_flat == majority_class).mean())
    print(f'\n[{probe_name}] Baseline — majority class ({majority_class}): {majority_acc:.4f}')

    # Random-label baseline: train probe on shuffled labels (use middle layer).
    mid_layer = num_layers_plus_one // 2
    X_train_mid_raw = all_hidden[train_idx, mid_layer].reshape(-1, emb_dim)
    X_test_mid_raw = all_hidden[test_idx, mid_layer].reshape(-1, emb_dim)
    scaler_mid = StandardScaler()
    X_train_mid = scaler_mid.fit_transform(X_train_mid_raw)
    X_test_mid = scaler_mid.transform(X_test_mid_raw)
    rng_state = np.random.RandomState(0)
    y_train_shuffled = rng_state.permutation(y_train_flat)
    clf_rand = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', n_jobs=-1)
    clf_rand.fit(X_train_mid, y_train_shuffled)
    random_label_acc = float(accuracy_score(y_test_flat, clf_rand.predict(X_test_mid)))
    print(f'[{probe_name}] Baseline — random labels (layer {mid_layer}): {random_label_acc:.4f}')

    accuracies = []
    print(f'[{probe_name}] Training {num_layers_plus_one} linear probes...')
    for layer_idx in range(num_layers_plus_one):
        X_train_raw = all_hidden[train_idx, layer_idx].reshape(-1, emb_dim)
        X_test_raw = all_hidden[test_idx, layer_idx].reshape(-1, emb_dim)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', n_jobs=-1)
        clf.fit(X_train, y_train_flat)
        acc = accuracy_score(y_test_flat, clf.predict(X_test))
        accuracies.append(float(acc))

        label_name = 'embed' if layer_idx == 0 else f'L{layer_idx}'
        print(f'  Layer {label_name}: accuracy = {acc:.4f}')

    return accuracies, majority_acc, random_label_acc


def plot_results(accuracies, majority_acc, random_label_acc, output_path):
    """Saves the attack-map probe accuracy plot."""
    x = list(range(len(accuracies)))
    x_labels = ['emb'] + [str(i) for i in range(1, len(accuracies))]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, accuracies, marker='o', color='seagreen', linewidth=2,
            markersize=5, label='Probe accuracy')
    ax.axhline(y=majority_acc, color='darkorange', linestyle='--', linewidth=1.2,
               label=f'Majority class ({majority_acc:.2f})')
    ax.axhline(y=random_label_acc, color='gray', linestyle=':', linewidth=1.2,
               label=f'Random labels ({random_label_acc:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel('Layer (0 = embedding, 1–16 = transformer blocks)')
    ax.set_ylabel('Test accuracy')
    ax.set_title('Linear probe: attack map\n(BC-270M, ood_puzzles)')
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f'\nSaved figure to {output_path}')


def main():
    # Load model.
    print('Building predictor with hidden states...')
    predictor, params = build_bc_270m_predictor_with_hidden_states()
    print('Model loaded.')

    # Load positions from ood_puzzles.csv, pushing the first move to get the
    # actual puzzle position.
    df = pd.read_csv(PUZZLES_CSV, nrows=1000)
    fen_strings = []
    for _, row in df.iterrows():
        board = chess.Board(row['FEN'])
        board.push(chess.Move.from_uci(row['Moves'].split()[0]))
        fen_strings.append(board.fen())
    print(f'Loaded {len(fen_strings)} FEN positions from {PUZZLES_CSV}')

    # Extract hidden states.
    all_hidden = extract_hidden_states(predictor, params, fen_strings)
    print(f'Hidden states shape: {all_hidden.shape}')

    # Compute attack-map labels.
    print('\nComputing attack-map labels...')
    attack_labels = compute_attack_labels(fen_strings)
    flat = attack_labels.reshape(-1)
    for cls, name in enumerate(_ATTACK_LABEL_NAMES):
        print(f'  class {cls} ({name}): {(flat == cls).sum()} squares '
              f'({100 * (flat == cls).mean():.1f}%)')

    # Train attack-map probes.
    accuracies, majority_acc, random_label_acc = train_probes(
        all_hidden, attack_labels, probe_name='attack-map'
    )

    # Plot.
    plot_results(accuracies, majority_acc, random_label_acc, OUTPUT_PDF)

    print('\n=== Summary ===')
    print(f'{"Layer":<8} {"Accuracy":>10}')
    for i, acc in enumerate(accuracies):
        tag = 'embed' if i == 0 else f'layer {i:2d}'
        print(f'{tag:<8} {acc:>10.4f}')
    print(f'{"majority":<8} {majority_acc:>10.4f}')
    print(f'{"rand-lbl":<8} {random_label_acc:>10.4f}')


if __name__ == '__main__':
    main()
