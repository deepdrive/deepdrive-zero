import os

from spinup.utils.test_policy import load_policy, run_policy

# Register custom envs
import gym_match_input_continuous
import deepdrive_zero
import gym

TEST_STATIC_OBSTACLE = False



if TEST_STATIC_OBSTACLE:
    _, get_action = load_policy(
        '/home/c2/src/spinningup/data/dd2d-ppo-intersection/dd2d-ppo-intersection_s0',
        use_model_only=False)

    env = gym.make('deepdrive-2d-static-obstacle-no-g-pen-v0')
else:
    _, get_action = load_policy(
        '/home/c2/src/spinningup/data/dd2d-ppo-intersection-g-lane-decel-fine-tune/dd2d-ppo-intersection-g-lane-decel-fine-tune_s0',
        use_model_only=False, deterministic=True)
    # env = gym.make('deepdrive-2d-intersection-v0')
    env = gym.make('deepdrive-2d-intersection-w-gs-allow-decel-v0')

# env.unwrapped.physics_steps_per_observation = 1
run_policy(env, get_action)