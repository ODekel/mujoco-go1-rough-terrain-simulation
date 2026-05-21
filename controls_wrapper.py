import logging
from collections.abc import Callable

import numpy as np
from jax.typing import ArrayLike

from controls import SetCommand


logger = logging.getLogger(__name__)


class ControlsWrapper:
    def __init__(self, controls_handler: Callable[[SetCommand], None]):
        self.controls_handler = controls_handler
        self._command = np.array([0.0, 0.0, 0.0])

    def _set_command(self, x: float, y: float, yaw: float) -> None:
        self._command[0] = y
        self._command[1] = x
        self._command[2] = yaw

    def handle(self) -> ArrayLike:
        self.controls_handler(self._set_command)
        logger.debug(f'Current command: {self._command}')
        return self._command
