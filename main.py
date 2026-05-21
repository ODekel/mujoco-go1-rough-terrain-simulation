import json
import logging
import sys

from controls import setup
from controls_wrapper import ControlsWrapper
from environment import load_jax_environment
from inference import create_inference_func
from random_wrapper import RandomWrapper
from simulator import simulate
from viewer_wrapper import run_viewer

if __name__ == '__main__':
    with open('./config.json', 'r') as f:
        config = json.load(f)
    logging.basicConfig(stream=sys.stdout, level=config['logging']['level'], force=True)
    control_handler = setup(config['controls'])
    env = load_jax_environment(config['env'])
    infer = create_inference_func(config['policy'])
    simulate(env, infer, ControlsWrapper(control_handler).handle, run_viewer, RandomWrapper().get_key)
