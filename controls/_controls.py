from collections.abc import Callable
from typing import Any

SetCommand = Callable[[float, float, float], None]
SetupCallable = Callable[[Any], Callable[[SetCommand], None]]
