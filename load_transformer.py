import os
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src.engines import neural_engines
from searchless_chess.src import training_utils
from searchless_chess.src import tokenizer
from key_mapping import key_mapping_function
from jax import random as jrandom
import numpy as np

#to get rid of the warnings: W external/xla/xla/service/gpu/autotuning/dot_search_space.cc:200] All configs were filtered out because none of them sufficiently match the hints. Maybe the hints set does not contain a good representative set of valid configs?Working around this by using the full hints set instead.
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"


def load_transformer_model(model_name):
    if model_name == "BC":
        # Configure the transformer model
        predictor_config = transformer.TransformerConfig(
            vocab_size=len(tokenizer._CHARACTERS),
            output_size=utils.NUM_ACTIONS,  # For behavioral cloning, output size is number of actions
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=tokenizer.SEQUENCE_LENGTH + 1,  # BC uses +1 instead of +2
            num_heads=8,
            num_layers=8, 
            embedding_dim=256, 
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )
        
        predictor = transformer.build_transformer_predictor(config=predictor_config)
        dummy_params = predictor.initial_params(
            rng=jrandom.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        )
        #for k, v in dummy_params.items():
        #    for k2, v2 in v.items():
       #         print(f"{k}.{k2}: {v2.shape}")
        # Load the trained parameters using key_mapping
        params = key_mapping_function(model_name)
        

        # Create prediction function
        predict_fn = neural_engines.wrap_predict_fn(predictor, params, batch_size=1)
        # Create the neural engine
        neural_engine = neural_engines.BCEngine(
            predict_fn=predict_fn,
            #temperature=0.005,  # Add some randomness to the moves
        )
        return neural_engine
    elif model_name == "BC 270M":
        # Configure the transformer model
        predictor_config = transformer.TransformerConfig(
            vocab_size=len(tokenizer._CHARACTERS),
            output_size=utils.NUM_ACTIONS,  # For behavioral cloning, output size is number of actions
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=tokenizer.SEQUENCE_LENGTH + 1,  # BC uses +1 instead of +2
            num_heads=8,
            num_layers=16, 
            embedding_dim=1024, 
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )
        
        predictor = transformer.build_transformer_predictor(config=predictor_config)
        checkpoint_dir = os.path.join(os.getcwd(), 'searchless_chess/checkpoints/local/behavioral_cloning')
        dummy_params = predictor.initial_params(
            rng=jrandom.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        )
        #for k, v in dummy_params.items():
        #    for k2, v2 in v.items():
       #         print(f"{k}.{k2}: {v2.shape}")
        # Load the trained parameters using key_mapping
        params = training_utils.load_parameters(
            checkpoint_dir=checkpoint_dir,
            params=dummy_params,
            step=200000
        )
        

        # Create prediction function
        predictor = transformer.build_transformer_predictor(config=predictor_config)
        predict_fn = neural_engines.wrap_predict_fn(predictor, params, batch_size=1)
        # Create the neural engine
        neural_engine = neural_engines.BCEngine(
            predict_fn=predict_fn,
            #temperature=0.005,  # Add some randomness to the moves
        )
        return neural_engine
    elif model_name == "270M":
        # Action value model settings
        num_return_buckets = 128  # Set to the value used during training
        _, return_buckets_values = utils.get_uniform_buckets_edges_values(num_return_buckets)
        
        predictor_config = transformer.TransformerConfig(
            vocab_size=utils.NUM_ACTIONS,
            output_size=num_return_buckets,  # Action value model: output is number of buckets
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
            num_heads=8,
            num_layers=16,
            embedding_dim=1024,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )
        
        predictor = transformer.build_transformer_predictor(config=predictor_config)
        checkpoint_dir = os.path.join(os.getcwd(), 'searchless_chess/checkpoints/270M')
        dummy_params = predictor.initial_params(
            rng=jrandom.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        )
        params = training_utils.load_parameters(
            checkpoint_dir=checkpoint_dir,
            params=dummy_params,
            step=6400000
        )
        
        predict_fn = neural_engines.wrap_predict_fn(predictor, params, batch_size=1)
        neural_engine = neural_engines.ActionValueEngine(
            return_buckets_values=return_buckets_values,
            predict_fn=predict_fn,
            #temperature=0.005,
        )
        return neural_engine
    elif model_name == "9M":
        # Action value model settings
        num_return_buckets = 128  # Set to the value used during training
        _, return_buckets_values = utils.get_uniform_buckets_edges_values(num_return_buckets)
        
        predictor_config = transformer.TransformerConfig(
            vocab_size=utils.NUM_ACTIONS,
            output_size=num_return_buckets,  # Action value model: output is number of buckets
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
            num_heads=8,
            num_layers=8,
            embedding_dim=256,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )
        
        predictor = transformer.build_transformer_predictor(config=predictor_config)
        checkpoint_dir = os.path.join(os.getcwd(), 'searchless_chess/checkpoints/9M')
        dummy_params = predictor.initial_params(
            rng=jrandom.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        )

        params = training_utils.load_parameters(
            checkpoint_dir=checkpoint_dir,
            params=dummy_params,
            step=6400000
        )


        predict_fn = neural_engines.wrap_predict_fn(predictor, params, batch_size=1)
        neural_engine = neural_engines.ActionValueEngine(
            return_buckets_values=return_buckets_values,
            predict_fn=predict_fn,
            #temperature=0.005,
        )
        return neural_engine
    elif model_name == "SV":
        # State value model settings
        num_return_buckets = 128  # Set to the value used during training
        _, return_buckets_values = utils.get_uniform_buckets_edges_values(num_return_buckets)
        
        predictor_config = transformer.TransformerConfig(
            vocab_size=len(tokenizer._CHARACTERS),
            output_size=num_return_buckets,  # State value model: output is number of buckets
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=tokenizer.SEQUENCE_LENGTH + 1,
            num_heads=8,
            num_layers=8,
            embedding_dim=256,
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )
        
        predictor = transformer.build_transformer_predictor(config=predictor_config)
        dummy_params = predictor.initial_params(
            rng=jrandom.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        )
        params = key_mapping_function(model_name)


        predict_fn = neural_engines.wrap_predict_fn(predictor, params, batch_size=1)
        neural_engine = neural_engines.StateValueEngine(
            return_buckets_values=return_buckets_values,
            predict_fn=predict_fn,
            #temperature=0.005,
        )
        return neural_engine
    else:
        raise NotImplementedError("This model does not exist or not implemented")
