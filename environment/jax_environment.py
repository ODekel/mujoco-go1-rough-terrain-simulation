import jax
from jax import Array
from mujoco import MjModel
from mujoco_playground import MjxEnv, State

from environment.environment import Environment, EnvironmentLoadConfig, _load_environment


class JaxEnvironment(Environment):
    def __init__(self, env: MjxEnv):
        self._reset = jax.jit(env.reset)
        self._step = jax.jit(env.step)
        self._model = env.mj_model

    def reset(self, key: Array) -> State:
        return self._reset(key)

    def step(self, state: State, action: Array) -> State:
        return self._step(state, action)

    @property
    def model(self) -> MjModel:
        return self._model


def load_jax_environment(config: EnvironmentLoadConfig) -> JaxEnvironment:
    return JaxEnvironment(_load_environment(config))
