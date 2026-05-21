import logging
from abc import ABC, abstractmethod
from typing import TypedDict, Required

import numpy as np
from PIL import Image
from jax import Array
from jax import numpy as jnp
from mujoco import MjModel
from mujoco_playground import State, MjxEnv, registry


_logger = logging.getLogger(__name__)


class Environment(ABC):
    @abstractmethod
    def reset(self, key: Array) -> State:
        pass

    @abstractmethod
    def step(self, state: State, action: Array) -> State:
        pass

    @property
    @abstractmethod
    def model(self) -> MjModel:
        pass


EnvironmentLoadConfig = TypedDict('EnvironmentLoadConfig', {
    'baseEnvName': str,
    'heightfieldOverrideImage': Required[str | None],
    'heightfieldScaling': float
})


def _load_environment(config: EnvironmentLoadConfig) -> MjxEnv:
    _logger.debug(f'Loading environment: {config['baseEnvName']}')
    env_cfg = registry.get_default_config(config['baseEnvName'])
    env_cfg['impl'] = 'jax'
    env = registry.load(config['baseEnvName'], config=env_cfg)

    if config['heightfieldOverrideImage']:
        _logger.debug(f'Overriding heightfield with image: {config['heightfieldOverrideImage']}')
        img = Image.open(str(config['heightfieldOverrideImage']))
        img_resized = img.resize((env.unwrapped.mjx_model.hfield_ncol[0], env.unwrapped.mjx_model.hfield_nrow[0]),
                                 Image.Resampling.BILINEAR)

        terrain_arr = jnp.array(img_resized).astype(jnp.float32) / 255.0
        new_hfield_data = jnp.array(terrain_arr.flatten())

        env.mj_model.hfield_data[:] = jnp.array(new_hfield_data)
        env.unwrapped._mjx_model = env.unwrapped.mjx_model.tree_replace({
            'hfield_data': new_hfield_data
        })

    _logger.debug(f'Scaling heightfield: {config["heightfieldScaling"]}')
    env.mj_model.hfield_size[0, 2] *= config['heightfieldScaling']
    new_hfield_size = np.copy(env.unwrapped.mjx_model.hfield_size)
    new_hfield_size[0, 2] *= config['heightfieldScaling']
    env.unwrapped._mjx_model = env.unwrapped.mjx_model.tree_replace({
        'hfield_size': new_hfield_size
    })

    return env
