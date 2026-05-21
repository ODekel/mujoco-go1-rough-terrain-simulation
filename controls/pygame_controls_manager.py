import functools
from typing import TypedDict

import pygame

from controls._controls import SetCommand, SetupCallable
from controls.pygame_controller_controls import PygameControllerConfig, PygameControllerControls
from controls.keyboard_controls import KeyboardControls

PygameControlsManagerConfig = TypedDict('PygameControlsManagerConfig', {'controller': PygameControllerConfig})


class PygameControlsManager:
    def __init__(self, config: PygameControlsManagerConfig):
        pygame.display.init()
        pygame.joystick.init()
        pygame.event.set_blocked(None)
        self._controller_controls = PygameControllerControls(config['controller'])
        self._keyboard_controls = KeyboardControls()

    def __call__(self, set_command: SetCommand):
        if self._controller_controls.is_ready():
            self._controller_controls(set_command)
            self._keyboard_controls(lambda x, y, yaw: None)
        else:
            self._controller_controls(lambda x, y, yaw: None)
            self._keyboard_controls(set_command)


setup: SetupCallable = functools.partial(PygameControlsManager)
