import chess
import os
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import neural_engines
from searchless_chess.src import training_utils
from searchless_chess.src import tokenizer
from jax import random as jrandom
import numpy as np

def load_transformer_model():
    # Configure the transformer model
    predictor_config = transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS,
        output_size=utils.NUM_ACTIONS,  # For behavioral cloning, output size is number of actions
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 1,  # BC uses +1 instead of +2
        num_heads=8,
        num_layers=12,  # 9M model has 12 layers
        embedding_dim=768,  # 9M model has 768 embedding dim
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )
    
    # Build the predictor
    predictor = transformer.build_transformer_predictor(config=predictor_config)
    
    # Set up checkpoint directory
    checkpoint_dir = os.path.join(os.getcwd(), 'searchless_chess/checkpoints/9M_behavioral_cloning')
    
    # Initialize parameters
    dummy_params = predictor.initial_params(
        rng=jrandom.PRNGKey(1),
        targets=np.ones((1, 1), dtype=np.uint32),
    )
    
    # Load the trained parameters
    params = training_utils.load_parameters(
        checkpoint_dir=checkpoint_dir,
        params=dummy_params,
        step=0,  # Use default to load the largest available step
    )
    
    # Create prediction function
    predict_fn = neural_engines.wrap_predict_fn(predictor, params, batch_size=1)
    
    # Create the neural engine
    neural_engine = neural_engines.BCEngine(
        predict_fn=predict_fn,
        temperature=0.005,  # Add some randomness to the moves
    )
    
    return neural_engine

def main():
    # Initialize the board
    fen = "rnbqkbnr/pppppppr/8/8/8/8/PPPPPPPQ/RNBQKBNN b" # KQkq - 0 1"  # Starting position
    board = chess.Board(fen)
    
    # Load the transformer model
    neural_engine = load_transformer_model()
    
    # Game loop
    while not board.is_game_over():
        # Print the current board
        print("\nCurrent board:")
        print(board)
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        print("\nLegal moves:", [move.uci() for move in legal_moves])
        
        # Get user's move
        while True:
            try:
                user_move = input("\nEnter your move (e.g., 'e2e4'): ")
                move = chess.Move.from_uci(user_move)
                if move in legal_moves:
                    break
                else:
                    print("Illegal move. Please try again.")
            except ValueError:
                print("Invalid move format. Please use format 'e2e4'.")
        
        # Make user's move
        board.push(move)
        
        # Check if game is over after user's move
        if board.is_game_over():
            break
        
        # Get transformer's move
        transformer_move = neural_engine.play(board)
        print(f"\nTransformer's move: {transformer_move}")
        
        # Make transformer's move
        board.push(transformer_move)
    
    # Print final board and game result
    print("\nFinal board:")
    print(board)
    print("\nGame result:", board.outcome().result())

if __name__ == "__main__":
    main() 