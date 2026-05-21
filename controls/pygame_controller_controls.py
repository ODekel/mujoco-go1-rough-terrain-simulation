import logging
from typing import TypedDict

import pygame
from pygame.event import Event

from controls._controls import SetCommand


_logger = logging.getLogger(__name__)


PygameControllerConfig = TypedDict('PygameControllerConfig', {
    'yawAxis': int,
    'deadzone': float
})


class PygameControllerControls:
    def __init__(self, config: PygameControllerConfig):
        pygame.event.set_allowed([pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED, pygame.JOYAXISMOTION])

        self._joysticks = {}
        self._config = config
        self._x, self._y, self._yaw = 0.0, 0.0, 0.0

    def _handle_joystick_motion(self, event: Event):
        val = event.value if abs(event.value) > self._config['deadzone'] else 0.0

        if event.axis == 0:
            self._x = -val
        elif event.axis == 1:
            self._y = -val
        elif event.axis == self._config['yawAxis']:
            self._yaw = -val
        else:
            joy = self._joysticks.get(event.instance_id)
            if joy:
                joy_name = joy.get_name()
            else:
                joy_name = 'UNKNOWN'
            _logger.debug(f'Event on unmapped axis: {event.axis}, '
                          f'joystick: {joy_name}')

    def is_ready(self) -> bool:
        return bool(self._joysticks)

    def __call__(self, set_command: SetCommand):
        for event in pygame.event.get([pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED, pygame.JOYAXISMOTION]):
            if event.type == pygame.JOYDEVICEADDED:
                joy = pygame.joystick.Joystick(event.device_index)
                self._joysticks[joy.get_instance_id()] = joy
                _logger.info(f"Controller connected: {joy.get_name()}")
            elif event.type == pygame.JOYDEVICEREMOVED:
                if event.instance_id in self._joysticks:
                    _logger.info(f"Controller disconnected: {self._joysticks[event.instance_id].get_name()}")
                    del self._joysticks[event.instance_id]
            elif event.type == pygame.JOYAXISMOTION:
                self._handle_joystick_motion(event)

        set_command(self._x, self._y, self._yaw)
