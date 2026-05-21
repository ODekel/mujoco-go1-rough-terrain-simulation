import keyboard

from controls._controls import SetCommand


_KEYBOARD_ACTIONS = {
    'left': lambda x, y, yaw: (x + 1.0, y, yaw),
    'right': lambda x, y, yaw: (x - 1.0, y, yaw),
    'up': lambda x, y, yaw: (x, y + 1.0, yaw),
    'down': lambda x, y, yaw: (x, y - 1.0, yaw),
    'comma': lambda x, y, yaw: (x, y, yaw + 1.0),
    'period': lambda x, y, yaw: (x, y, yaw - 1.0),
}


class KeyboardControls:
    def __call__(self, set_command: SetCommand):
        x, y, yaw = 0.0, 0.0, 0.0
        for key, action in _KEYBOARD_ACTIONS.items():
            if keyboard.is_pressed(key):
                x, y, yaw = action(x, y, yaw)
        set_command(x, y, yaw)
