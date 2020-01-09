from spinup.utils.test_policy import load_policy, run_policy

# Register custom envs
import gym_match_input_continuous
import deepdrive_2d
import gym

_, get_action = load_policy('/home/c2/src/spinningup/data/dd2d-ppo-static-obstacle-front-dist-no-gs/dd2d-ppo-static-obstacle-front-dist-no-gs_s0')
env = gym.make('deepdrive-2d-static-obstacle-v0')
env.unwrapped.enable_render()
# env.unwrapped.physics_steps_per_observation = 1
run_policy(env, get_action)