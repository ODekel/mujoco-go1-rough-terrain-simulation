import functools
from collections.abc import Callable

from brax.training.types import Policy
from jax import Array
from jax.typing import ArrayLike
from mujoco import mjx, MjData, MjModel
from mujoco_playground import State

from environment import Environment


_ControlsHandler = Callable[[], ArrayLike]


def _sync_data(data: MjData, model: MjModel, state: State):
    mjx.get_data_into(data, model, state.data)


def _calculate_single_step(state: State, env: Environment, infer: Policy, controls_handler: _ControlsHandler,
                           sync_data: Callable[[State], None], rng: Callable[[], Array]) -> State:
    command = controls_handler()
    state.info['command'] = command

    action, _ = infer(state.obs, rng())
    new_state = env.step(state, action)
    sync_data(state)

    return new_state


class Simulator:
    def __init__(self, env: Environment, infer: Policy, controls_handler: _ControlsHandler, rng: Callable[[], Array]
                 ) -> None:
        self._env = env
        self._model = env.model
        self._data = MjData(self._model)
        self._state = env.reset(rng())
        self._infer = infer
        self._controls_handler = controls_handler
        self._rng = rng

    def _single_step(self) -> None:
        self._state = _calculate_single_step(self._state, self._env, self._infer, self._controls_handler,
                                             functools.partial(_sync_data, self._data, self._model), self._rng)

    def simulate(self, sim_runner: Callable[[MjModel, MjData, Callable[[], None]], None]) -> None:
        sim_runner(self._model, self._data, self._single_step)


def simulate(env: Environment, infer: Policy, controls_handler: _ControlsHandler,
             sim_runner: Callable[[MjModel, MjData, Callable[[], None]], None], rng: Callable[[], Array]) -> None:
    Simulator(env, infer, controls_handler, rng).simulate(sim_runner)
