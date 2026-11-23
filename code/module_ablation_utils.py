"""Matched initial weights for the explicitly defined whole-module ablations."""

import hashlib

import torch

from OurModels.TDFormer import TDFormer


VARIANTS = ('full', 'wo_tan', 'wo_fan', 'wo_van')
TRUNK_PREFIXES = (
    'lstm_encoder.', 'lstm_decoder.', 'post_lstm_', 'multihead_attn.',
    'post_attn_gate_norm.', 'pos_wise_ff.', 'pre_output_gate_norm.', 'output_layer.',
)


def state_hash(state, names=None):
    digest = hashlib.sha256()
    for name in state if names is None else names:
        digest.update(name.encode('utf-8'))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_matched_model(dataset, model_kwargs, variant):
    """Keep Full's random initialization and copy every compatible tensor.

    A removed module necessarily changes keys and parameter counts. Only the
    new flat VAN adapter receives its own initialization. Constructing an
    ablated model does not advance the training RNG beyond Full's state.
    """
    if variant not in VARIANTS:
        raise ValueError(variant)
    kwargs = dict(model_kwargs)
    kwargs.update(use_variable_attention=True, use_feature_attention=True, use_temporal_attention=True)
    reference = TDFormer.from_dataset(dataset, module_ablation='full', **kwargs)
    reference_state = reference.state_dict()
    reference_hash = state_hash(reference_state)
    if variant == 'full':
        model = reference
    else:
        with torch.random.fork_rng(devices=[]):
            model = TDFormer.from_dataset(dataset, module_ablation=variant, **kwargs)
        actual_state = model.state_dict()
        for name, value in actual_state.items():
            if name in reference_state and value.shape == reference_state[name].shape:
                actual_state[name] = reference_state[name].detach().clone()
        model.load_state_dict(actual_state, strict=True)

    actual_state = model.state_dict()
    shared_names = [name for name in actual_state
                    if name in reference_state and actual_state[name].shape == reference_state[name].shape]
    assert all(torch.equal(actual_state[name], reference_state[name]) for name in shared_names)
    trunk_names = [name for name in reference_state if name.startswith(TRUNK_PREFIXES)]
    assert all(name in actual_state for name in trunk_names)
    assert all(torch.equal(actual_state[name], reference_state[name]) for name in trunk_names)
    audit = {
        'reference_full_sha256': reference_hash,
        'shared_trunk_sha256': state_hash(actual_state, trunk_names),
        'shared_tensor_count': len(shared_names),
        'new_tensor_names': [name for name in actual_state if name not in reference_state],
        'removed_tensor_names': [name for name in reference_state if name not in actual_state],
        'initialization_policy': 'copy all matching Full tensors; restore training RNG after adapter construction',
    }
    return model, audit
