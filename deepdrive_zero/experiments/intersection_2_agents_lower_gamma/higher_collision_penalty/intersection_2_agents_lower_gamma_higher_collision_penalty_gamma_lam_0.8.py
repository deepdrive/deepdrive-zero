import os
import sys

from deepdrive_zero.experiments import utils
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

experiment_name = os.path.basename(__file__)[:-3]
notes = """
Trying lower gamma and lam
"""


env_config = dict(
    env_name='deepdrive-2d-intersection-w-gs-allow-decel-v0',
    is_intersection_map=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=0,
    gforce_penalty_coeff=0,
    gforce_threshold=None,
    incent_win=True,
    constrain_controls=False,
    collision_penalty_coeff=1,
)

def train():
    eg = ExperimentGrid(name=experiment_name)
    eg.add('env_name', env_config['env_name'], '', False)
    # eg.add('seed', 0)
    eg.add('epochs', 8000)
    eg.add('gamma', 0.8)
    eg.add('lam', 0.8)
    # eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', (256, 256), 'hid')
    eg.add('ac_kwargs:activation', torch.nn.Tanh, '')
    eg.add('notes', notes, '')
    eg.add('run_filename', os.path.realpath(__file__), '')
    eg.add('env_config', env_config, '')
    eg.run(ppo_pytorch)


if __name__ == '__main__':
    utils.run(train_fn=train, env_config=env_config)