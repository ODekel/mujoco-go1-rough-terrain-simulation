import functools
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import jax
from brax.training.agents.ppo.checkpoint import load_policy
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Policy


InferenceConfig = TypedDict('InferenceConfig', {
    'savedPolicyPath': str,
    'policyHiddenLayerSizes': Sequence[int],
    'valueHiddenLayerSizes': Sequence[int]
})


def create_inference_func(config: InferenceConfig) -> Policy:
    network_factory = functools.partial(ppo_networks.make_ppo_networks,
                                        policy_hidden_layer_sizes=config['policyHiddenLayerSizes'],
                                        value_hidden_layer_sizes=config['valueHiddenLayerSizes'])
    make_inference = load_policy(Path(config['savedPolicyPath']).resolve(), network_factory, deterministic=True)
    jit_inference = jax.jit(make_inference)
    return jit_inference
