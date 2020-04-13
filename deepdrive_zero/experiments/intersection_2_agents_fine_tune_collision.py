import os
import sys

from deepdrive_zero.experiments import utils
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

experiment_name = 'deepdrive-2d-intersection-no-constrained-controls-example'
notes = """Fine tuning best model with a higher collision penalty. 
From now on will be saving state dicts with adam state, hopefully, not
just weights"""

env_config = dict(
    env_name='deepdrive-2d-intersection-w-gs-allow-decel-v0',
    is_intersection_map=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=0,
    gforce_penalty_coeff=0,
    collision_penalty_coeff=1,
    gforce_threshold=None,
    incent_win=True,
    constrain_controls=False,
)

def train():
    eg = ExperimentGrid(name=experiment_name)
    eg.add('env_name', env_config['env_name'], '', False)
    # eg.add('seed', 0)
    eg.add('resume', '/home/c2/src/tmp/spinningup/data/deepdrive-2d-intersection-no-constrained-controls-example/deepdrive-2d-intersection-no-constrained-controls-example_s0_2020_03-10_13-14.50/best_HorizonReturn/2020_03-11_11-36.27')
    eg.add('reinitialize_optimizer_on_resume', False)
    eg.add('pi_lr', 3e-5)  # doesn't seem to have an effect, but playing it safe and lowering learning rate since we're not restoring adam rates
    eg.add('vf_lr', 1e-4)  # doesn't seem to have an effect, but playing it safe and lowering learning rate since we're not restoring adam rates
    eg.add('epochs', 8000)
    # eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', (256, 256), 'hid')
    eg.add('ac_kwargs:activation', torch.nn.Tanh, '')
    eg.add('notes', notes, '')
    eg.add('run_filename', os.path.realpath(__file__), '')
    eg.add('env_config', env_config, '')
    eg.run(ppo_pytorch)


if __name__ == '__main__':
    utils.run(train_fn=train, env_config=env_config)