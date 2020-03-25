import os
import sys

from deepdrive_zero.experiments import utils
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

experiment_name = 'deepdrive-2d-intersection-no-constrained-controls-example'
notes = 'Using new type of experiment running method.'

env_config = dict(
    env_name='deepdrive-2d-intersection-w-gs-allow-decel-v0',
    is_intersection_map=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=0,
    gforce_penalty_coeff=0,
    end_on_harmful_gs=False,
    incent_win=True,
    constrain_controls=False,
)

net_config = dict(
    hidden_units=(256, 256),
    activation=torch.nn.Tanh
)

def train():
    eg = ExperimentGrid(name=experiment_name)
    eg.add('env_name', env_config['env_name'], '', False)
    # eg.add('seed', 0)
    eg.add('epochs', 8000)
    # eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', (256, 256), 'hid')
    eg.add('ac_kwargs:activation', torch.nn.Tanh, '')
    eg.add('notes', notes, '')
    eg.add('run_filename', os.path.realpath(__file__), '')
    eg.add('env_config', env_config, '')
    eg.run(ppo_pytorch)


if __name__ == '__main__':
    utils.run(train_fn=train, env_config=env_config, net_config=net_config)