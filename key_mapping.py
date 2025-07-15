import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
import json
import ast
import ast
from flax import traverse_util

def key_mapping_function(model_name):
    if model_name == "BC":
        folder_name = "9M_behavioral_cloning"
    elif model_name == "SV":
        folder_name = "9M_state_value"

    # Path to the params directory and metadata file
    ckpt_path = "/home/am3049/ood_chess/searchless_chess/checkpoints/" + folder_name
    metadata_path = ckpt_path + "/_METADATA"

    # Parse the _METADATA file to get the parameter tree structure
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    tree_metadata = metadata["tree_metadata"]

    #Build a nested dictionary from the tuple keys in the metadata
    def nested_dict():
        # Convert string keys to actual tuple keys
        flat_dict = {ast.literal_eval(k): v for k, v in tree_metadata.items()}
        # Unflatten the dictionary
        nested_dict = traverse_util.unflatten_dict(flat_dict)
        return nested_dict 

    param_tree = nested_dict()

    def insert(tree, key_tuple):
        d = tree
        for k in key_tuple[:-1]:
            d = d[k]
        d[key_tuple[-1]] = None  # Placeholder for ArrayRestoreArgs

    for k in tree_metadata.keys():
        key_tuple = ast.literal_eval(k)
        insert(param_tree, key_tuple)

    # Fill the leaves with ArrayRestoreArgs using single-device sharding
    device = jax.devices()[0]
    sharding = jax.sharding.SingleDeviceSharding(device)

    def fill_restore_args(tree):
        if isinstance(tree, dict):
            return {k: fill_restore_args(v) for k, v in tree.items()}
        else:
            return ocp.ArrayRestoreArgs(sharding=sharding, dtype=jnp.float32)

    restore_args = fill_restore_args(param_tree)

    # Restore the checkpoint using these restore_args
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_path, restore_args=restore_args)
    params = restored 

    #Flatten the nested dictionary keys
    def flatten_dict(d, parent_key='', sep='/'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_params = flatten_dict(params)

    #for k, v in flat_params.items():
    #    print(f"{k}: {v.shape}")

    # Define and apply a key mapping function
    def mapping_func(old_key):
        new_key = old_key.replace('MultiHeadDotProductAttention_', 'multi_head_dot_product_attention_')
        new_key = new_key.replace('Dense_', 'linear_')
        new_key = new_key.replace('Embed_', 'embed_')
        new_key = new_key.replace('LayerNorm_', 'layer_norm_')
        new_key = new_key.replace('key/kernel', 'linear_1/w')
        new_key = new_key.replace('out/kernel', 'linear_3/w')
        new_key = new_key.replace('query/kernel', 'linear/w')
        new_key = new_key.replace('value/kernel', 'linear_2/w')
        new_key = new_key.replace('embedding/value', 'embeddings')
        new_key = new_key.replace('bias', 'offset')
        new_key = new_key.replace('kernel/value', 'w')
        new_key = new_key.replace('linear_24/offset','linear_24/b') 
        new_key = new_key.replace('_0', '')
        new_key = new_key.replace('params/', '')
        return new_key

    remapped_params = {mapping_func(k): v for k, v in flat_params.items()}

    
    #for k, v in remapped_params.items():
    #    print(f"{k}: {v.shape}")

    #reshape the attention layers from (256, 8, 32), (8, 32, 256) to (256, 256)
    for k, v in remapped_params.items():
        if v.shape == (256, 8, 32):
            remapped_params[k] = v.reshape(256, 256)
        if v.shape == (8, 32, 256):
            remapped_params[k] = v.reshape(256, 256)

    
    #for k, v in remapped_params.items():
    #    print(f"{k}: {v.shape}")

    #Making a 2-level dictionary to match the model's expected structure
    def partial_unflatten_dict(flat_dict, sep='/'):
        result = {}
        for k, v in flat_dict.items():
            if sep in k:
                prefix, last = k.rsplit(sep, 1)
                if prefix not in result:
                    result[prefix] = {}
                result[prefix][last] = v
        return result

    partially_unflattened_params = partial_unflatten_dict(remapped_params)

    #for k, v in partially_unflattened_params.items():
    #    for k2, v2 in v.items():
    #        print(f"{k}.{k2}: {v2.shape}")
    
    return partially_unflattened_params
