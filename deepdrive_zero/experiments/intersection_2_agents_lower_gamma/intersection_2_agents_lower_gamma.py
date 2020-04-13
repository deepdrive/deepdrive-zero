import os
import sys

from deepdrive_zero.experiments import utils
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

experiment_name = os.path.basename(__file__)[:-3]
notes = """
Try a smaller gamma to speed up learning of smaller grain interactions. Our 
reward is mostly per frame anyway, excepting for the win reward, so we could
likely turn this way down. Also lowering lambda for GAE to match the ratio of
effective horizons between GAE and value targets (gamma) where the default
we've been using is gamma=0.99 ~ 100 steps and lambda=0.97 ~ 33 steps. For us
that's a default of 10 seconds and 3.3 seconds. Here we'll try gamma=0.95
~ 2 seconds and lambda=0.835 ~ 600ms. 
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
)

def train():
    eg = ExperimentGrid(name=experiment_name)
    eg.add('env_name', env_config['env_name'], '', False)
    # eg.add('seed', 0)
    eg.add('epochs', 8000)
    eg.add('gamma', 0.95)
    eg.add('lam', 0.835)
    # eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', (256, 256), 'hid')
    eg.add('ac_kwargs:activation', torch.nn.Tanh, '')
    eg.add('notes', notes, '')
    eg.add('run_filename', os.path.realpath(__file__), '')
    eg.add('env_config', env_config, '')
    eg.run(ppo_pytorch)


if __name__ == '__main__':
    utils.run(train_fn=train, env_config=env_config)