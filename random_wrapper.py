import jax
from jax import Array


class RandomWrapper:
    def __init__(self) -> None:
        self._rng = jax.random.key(0)

    def get_key(self) -> Array:
        rng, key = jax.random.split(self._rng)
        self._rng = rng
        return key
