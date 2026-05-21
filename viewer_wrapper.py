import logging
import time
from collections.abc import Callable

from mujoco import viewer, MjData, MjModel


logger = logging.getLogger(__name__)


def run_viewer(model: MjModel, data: MjData, calculate_single_step: Callable[[], None]):
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            step_start = time.time()
            logger.debug(f"Next step start: {step_start}")

            calculate_single_step()

            v.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
