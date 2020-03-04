import os

from spinup.utils.test_policy import load_policy, run_policy

# Register custom envs
import gym_match_input_continuous
import deepdrive_zero
import gym

TEST_STATIC_OBSTACLE = False


# TODO: Move configs from env to python that train and test both use

if TEST_STATIC_OBSTACLE:
    _, get_action = load_policy(
        '/home/c2/src/spinningup/data/dd2d-ppo-intersection/dd2d-ppo-intersection_s0',
        use_model_only=False)

    env = gym.make('deepdrive-2d-static-obstacle-no-g-pen-v0')
else:
    p = '/home/c2/src/spinningup/data/dd0-ppo-one-waypoint-steer-only-delta-controls-no-g-pen-incent-win-cmn-inputs3/dd0-ppo-one-waypoint-steer-only-delta-controls-no-g-pen-incent-win-cmn-inputs3_s0_2020_03-03_13-30.40/best_HorizonReturn/2020_03-03_16-28.34'
    if 'no-end-g' in p or 'no-contraint-g' in p or 'no-g' in p:
        os.environ['END_ON_HARMFUL_GS'] = '0'
        os.environ['GFORCE_PENALTY_COEFF'] = '0'
        os.environ['JERK_PENALTY_COEFF'] = '0'
    if 'no-contraint' in p and 'w-constraint' not in p:
        os.environ['CONSTRAIN_CONTROLS'] = '0'
    if 'delta-controls' in p:
        os.environ['EXPECT_NORMALIZED_ACTION_DELTAS'] = '1'
    else:
        os.environ['EXPECT_NORMALIZED_ACTION_DELTAS'] = '0'

    if 'one-waypoint' in p:
        env_name = 'deepdrive-2d-one-waypoint-v0'
    else:
        env_name = 'deepdrive-2d-intersection-v0'
    _, get_action = load_policy(p, use_model_only=False, deterministic=True)
    # env = gym.make('deepdrive-2d-intersection-v0')
    env = gym.make(env_name)

# env.unwrapped.physics_steps_per_observation = 1
run_policy(env, get_action)
