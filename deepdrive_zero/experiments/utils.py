import sys

import gym

from spinup.utils.test_policy import load_policy_and_env, run_policy


def run(train_fn, env_config, net_config=None, try_rollouts=0,
        steps_per_try_rollout=0, num_eval_episodes=100):
    mode = '--train' if not sys.argv[1:] else sys.argv[1]
    if mode == '--train':
        train_fn()
    elif mode == '--play':
        from deepdrive_zero import player
        # env_config['physics_steps_per_observation'] = 1
        player.start(env_config=env_config)
    elif mode == '--test':
        if not sys.argv[2:]:
            model_path = get_latest_model_path()  # TODO
        else:
            model_path = sys.argv[2]
        env = gym.make(env_config['env_name'])
        env.configure_env(env_config)
        if try_rollouts != 0:
            deterministic = False
        else:
            deterministic = True
        _, get_action = load_policy_and_env(fpath=model_path,
                                            deterministic=deterministic,
                                            env=env, net_config=net_config)
        run_policy(env, get_action,
                   try_rollouts=try_rollouts,
                   steps_per_try_rollout=steps_per_try_rollout,
                   num_episodes=num_eval_episodes)