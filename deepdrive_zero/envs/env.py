import os
import sys
import time
from copy import deepcopy
from inspect import signature
from typing import Tuple, List
import random
import gym
import numpy as np
from box import Box
from gym import spaces
from .agent import get_closest_point

import pyglet

from deepdrive_zero.envs.agent import Agent
from deepdrive_zero.physics.collision_detection import check_collision_ego_obj,\
    check_collision_agents
from deepdrive_zero.constants import USE_VOYAGE, MAP_WIDTH_PX, MAP_HEIGHT_PX, \
    SCREEN_MARGIN, VEHICLE_LENGTH, VEHICLE_WIDTH, PX_PER_M, \
    MAX_METERS_PER_SEC_SQ, IS_DEBUG_MODE, GAME_OVER_PENALTY, FPS, \
    PARTIAL_PHYSICS_STEP, COMPLETE_PHYSICS_STEP
from deepdrive_zero.logs import log


class Deepdrive2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    is_deepdrive = True

    def __init__(self,
                 px_per_m=PX_PER_M,
                 add_rotational_friction=True,
                 add_longitudinal_friction=True,
                 return_observation_as_array=True,
                 seed_value=0,
                 expect_normalized_actions=True,
                 expect_normalized_action_deltas=False,
                 decouple_step_time=True,
                 physics_steps_per_observation=6,
                 is_one_waypoint_map=False,
                 is_intersection_map=False,
                 match_angle_only=False,
                 incent_win=False,
                 gamma=0.99,
                 add_static_obstacle=False,
                 disable_gforce_penalty=False,
                 forbid_deceleration=False,
                 contain_prev_actions_in_obs=True, #add prev actions into obs vector or not. False for r2d1
                 being_played=False,):

        self.logger = log

        log.info(f'{sys.executable} {sys.argv}')


        # Env config -----------------------------------------------------------
        self.env_config = dict(
            jerk_penalty_coeff=0.10,
            gforce_penalty_coeff=0.031,
            lane_penalty_coeff=0.02,
            collision_penalty_coeff=0.31,
            speed_reward_coeff=0.50,
            win_coefficient=1,
            gforce_threshold=1,
            jerk_threshold=None,
            constrain_controls=False,
            ignore_brake=False,
            forbid_deceleration=forbid_deceleration,
            expect_normalized_action_deltas=expect_normalized_action_deltas,
            discrete_actions=None,
            incent_win=incent_win,
            dummy_accel_agent_indices=None,
            wait_for_action=False,
            incent_yield_to_oncoming_traffic=False,
            physics_steps_per_observation=physics_steps_per_observation,
            end_on_lane_violation=False,
            contain_prev_actions_in_obs=contain_prev_actions_in_obs,
            dummy_random_scenario=False,
        )

        # All units in SI units (meters and radians) unless otherwise specified
        self.return_observation_as_array: bool = return_observation_as_array
        self.px_per_m: float = px_per_m
        self.expect_normalized_actions: bool = expect_normalized_actions
        self.seed_value: int = seed_value
        self.add_rotational_friction: bool = add_rotational_friction
        self.add_longitudinal_friction: bool = add_longitudinal_friction
        self.static_map: bool = '--static-map' in sys.argv
        self.disable_gforce_penalty = disable_gforce_penalty
        self.contain_prev_actions_in_obs = contain_prev_actions_in_obs


        # The previous observation, reward, done, info for each agent
        # Useful for running / training the agents
        self.agent_step_outputs = []  # TODO: Use pre-allocated numpy array here

        # For faster / slower than real-time stepping
        self.decouple_step_time = decouple_step_time

        self.fps: int = FPS
        self.target_dt: float = 1 / self.fps

        self.match_angle_only: bool = match_angle_only
        self.is_one_waypoint_map: bool = is_one_waypoint_map
        self.is_intersection_map: bool = is_intersection_map
        self.gamma: float = gamma
        self.add_static_obstacle: bool = add_static_obstacle

        # max_one_waypoint_mult
        # Specifies distance to waypoint as ratio: distance / map_size
        # 0.22 m/s on 0.1
        # Less than 2.5 m/s on 0.1?
        self.max_one_waypoint_mult = 0.5

        np.random.seed(self.seed_value)

        self.player = None

        self.should_render = False
        self._has_enabled_render = False

        if self.is_intersection_map:
            self.num_agents = 2
        else:
            self.num_agents = 1
        self.dummy_accel_agent_indices: List[int] = []

        self.agent_index: int = 0  # Current agent we are stepping
        self.discrete_actions = None
        self.being_played = being_played
        self.update_intermediate_physics = self.should_render or self.being_played
        self.render_choppy_but_realtime = False
        # End env config -------------------------------------------------------

        # Env state ------------------------------------------------------------
        # Step properties
        self.episode_steps: int = 0
        self.num_episodes: int = 0
        self.total_steps: int = 0
        self.last_step_time: float = None
        self.wall_dt: float = None
        self.last_sleep_time: float = None
        self.start_step_time: float = None
        self.total_episode_time: float = 0
        self.curr_reward = 0
        self.agents = None
        self.dummy_accel_agents = None
        self.all_agents = None  # agents + dummy_agents

        self.agent_index: int = 0  # Current agent we are stepping
        self.curr_reward = 0

        self.dummy_action = None
        self.last_step_output = None
        # End env state --------------------------------------------------------

    def get_state(self):
        return (self.episode_steps,
                self.num_episodes,
                self.total_steps,
                self.last_step_time,
                self.wall_dt,
                self.last_sleep_time,
                self.start_step_time,
                self.total_episode_time,
                self.curr_reward,
                [a.get_state() for a in self.all_agents],)

    def set_state(self, s):
        (self.episode_steps,
         self.num_episodes,
         self.total_steps,
         self.last_step_time,
         self.wall_dt,
         self.last_sleep_time,
         self.start_step_time,
         self.total_episode_time,
         self.curr_reward) = s[:-1]

        agent_states = s[-1]
        for i, agent in enumerate(self.all_agents):
            agent.set_state(agent_states[i])

    def configure_env(self, env_config: dict = None):
        env_config = self._set_config(env_config or {})
        env_config_box = Box(env_config, default_box=True)
        if env_config_box.is_intersection_map:
            self.is_intersection_map = env_config_box.is_intersection_map

        self.num_dummy_agents = len(self.env_config['dummy_accel_agent_indices']) if self.env_config['dummy_accel_agent_indices'] is not None else 0

        # Pass env config params to agent if they are arguments to agent
        # constructor. # TODO: Move to an agent section of the config.

        agent_params = signature(Agent).parameters.keys()
        agent_config = {k: v for k,v in self.env_config.items() if k in agent_params}
        self.agents: List[Agent] = [Agent(
                env=self,
                agent_index=i,
                disable_gforce_penalty=self.disable_gforce_penalty,
                **agent_config)
            for i in range(self.num_agents)]

        dummies = self.env_config['dummy_accel_agent_indices']
        if dummies is not None:
            self.dummy_accel_agent_indices = dummies

        #TODO: set dummy agent to continuous action space
        dummy_agent_config = agent_config
        dummy_agent_config['discrete_actions'] = None #to set continuous actions for dummy agent
        self.dummy_accel_agents: List[Agent] = [Agent(
                env=self,
                agent_index=i,
                disable_gforce_penalty=self.disable_gforce_penalty,
                **agent_config)
            for i in self.dummy_accel_agent_indices]

        self.all_agents = self.agents + self.dummy_accel_agents

        self.num_agents = len(self.agents)
        self.discrete_actions = self.env_config['discrete_actions']
        self.physics_steps_per_observation = env_config['physics_steps_per_observation']

        if '--no-timeout' in sys.argv:
            max_seconds = 100000
        elif '--one_waypoint_map' in sys.argv:
            self.is_one_waypoint_map = True
            max_seconds = self.max_one_waypoint_mult * 200
        elif self.is_intersection_map:
            max_seconds = 60
        else:
            max_seconds = 60

        self._max_episode_steps = (max_seconds *
                                   1/self.target_dt *
                                   1/self.physics_steps_per_observation)

        self.reset()
        self.setup_spaces()

    def _set_config(self, env_config):
        name_col_len = 45
        orig_config = deepcopy(self.env_config)
        self.env_config.update(env_config)
        for k, v in self.env_config.items():
            if k not in orig_config or orig_config[k] != v:
                name_col = f'{k.lower()}'
                custom = True
            else:
                name_col = f'{k.lower()}'
                custom = False
            padding = ' ' * (name_col_len - len(name_col))
            description = 'custom ' if custom else 'default'
            log.info(f'{name_col}{padding}{description} {v}')
        return self.env_config

    def setup_spaces(self):
        # Action space: ----
        # Steer, Accel, Brake
        agent = self.agents[0]
        if self.discrete_actions:
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        elif self.expect_normalized_actions:
            self.action_space = spaces.Box(low=-1, high=1, shape=(agent.num_actions,))
        else:
            # https://www.convert-me.com/en/convert/acceleration/ssixtymph_1.html?u=ssixtymph_1&v=7.4
            # Max voyage accel m/s/f = 3.625 * FPS = 217.5 m/s/f
            # TODO: Set steering limits as well
            self.action_space = spaces.Box(low=-10.2, high=10.2, shape=(agent.num_actions,))
        blank_obz = agent.get_blank_observation()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(blank_obz),))

    def _enable_render(self):
        from deepdrive_zero import player
        self.player = player.start(
            env=self,
            fps=self.fps)
        pyglet.app.event_loop.has_exit = False
        pyglet.app.event_loop._legacy_setup()
        pyglet.app.platform_event_loop.start()
        pyglet.app.event_loop.dispatch_event('on_enter')
        pyglet.app.event_loop.is_running = True

        self.should_render = True

    def reset(self):
        self.curr_reward = 0
        self.total_episode_time = 0
        self.dummy_action = [0, 1, -random.random()] # reset dummy agent action. In this way it will have a constant action in each episode
        if self.agent_step_outputs:
            for agent in self.dummy_accel_agents:
                if agent.done:
                    agent.reset()
            # Just reset the current agent and dummy agents
            return self.agents[self.agent_index].reset()
        else:
            # First reset, reset entire env
            self.episode_steps = 0

            for agent in self.agents:
                o, r, done, info = agent.reset(), 0, False, {}
                self.agent_step_outputs.append((o, r, done, info))

            for agent in self.dummy_accel_agents:
                agent.reset()

        return self.get_blank_observation()

    def seed(self, seed=None):
        self.seed_value = seed or 0
        random.seed(seed)

    @log.catch(reraise=True)
    def step(self, action):
        if self.total_steps == 0:
            log.info(self.env_config)
        self.start_step_time = time.time()
        agent = self.agents[self.agent_index] #select agent based on index- it will swith in every _step() call
        self.check_for_collisions()
        step_out = agent.step(action)
        if step_out == PARTIAL_PHYSICS_STEP:
            return step_out
        ret = self.finish_step()
        return ret

    def finish_step(self):
        agent = self.agents[self.agent_index]
        obs, reward, done, info = agent.last_step_output
        self.curr_reward = reward
        if done:
            self.num_episodes += 1
        self.episode_steps += 1
        self.total_steps += 1

        ret = self.get_step_output(done, info, obs, reward) # if len(self.agents)>1 -> one agent.step -> get the obs for other agent

        if self.should_render:
            self.regulate_fps()

        # one step for dummy agents
        for dummy_accel_agent in self.dummy_accel_agents:
            # Random forward accel
            # _, _, d, _ = dummy_accel_agent.step(self.dummy_action)

            # p-controller for steering
            steer = dummy_accel_agent.lateral_control()
            self.dummy_action[0] = steer
            _, _, d, _ = dummy_accel_agent.step(self.dummy_action)

            if d: #if done -> reset dummy agent
                dummy_accel_agent.reset()

        self.last_step_output = ret
        return ret

    def get_step_output(self, done, info, obs, reward):
        """ Return the observation that corresponds with the correct agent/action

        i.e. since we are looping through agents:

        agent_1_obs = reset()  # Get a blank observation, i.e. just zeroes
        agent_1_action = model(agent_1_obs)
        agent_2_obs = step(agent_1_action)  # step 1 - agent_2_obs is just blank
        agent_2_action = model(agent_2_obs)
        agent_1_obs = step(agent_2_action)  # step 2 - where agent_1_obs was from step 1 above
        agent_1_action = model(agent_1_obs)
        agent_2_obs = step(agent_1_action)  # step 3

        etc...

        This allows you to run the env the same as any other gym env
        in a step/reset loop.

        Just be sure to store states, actions, and rewards
        according to the env.agent_index as we do in PPOBuffer.

        NOTE: done and info are returned for the current agent, not the next
        agent, as those need to be acted on before querying the model.

        """
        agent_index = self.agent_index
        self.agent_step_outputs[agent_index] = (obs, reward, done, info)
        agent_index = self.total_steps % len(self.agents) # switch to other agent
        ret = self.agent_step_outputs[agent_index] # after one agent.step, get ret (obs, r, d, info) for the other agent.
        self.agent_index = agent_index
        return ret

    def regulate_fps(self):
        step_time = time.time() - self.start_step_time
        if self.should_render:
            target_dt = self.target_dt / self.num_agents
            if self.last_sleep_time is None:
                sleep_time = target_dt
                sleep_makeup = 0
            else:
                sleep_makeup = target_dt - step_time
                sleep_time = max(sleep_makeup, 0)
            time.sleep(sleep_time)
            self.last_sleep_time = sleep_time
            # final_step_time = time.time() - self.start_step_time
            # log.info(f'step time {final_step_time} slept {sleep_time} '
            #          f'sleep_makeup {sleep_makeup}')

    def get_dt(self):
        if self.last_step_time is not None:
            self.wall_dt = time.time() - self.last_step_time
        else:
            self.wall_dt = self.target_dt
        if self.decouple_step_time:
            dt = self.target_dt
        else:
            dt = self.wall_dt
        return dt

    def get_blank_observation(self):
        return self.agents[0].get_blank_observation()

    def render(self, mode='human'):
        if not self._has_enabled_render:
            self._enable_render()
            self._has_enabled_render = True
            # if self.physics_steps_per_observation != 1:
            #     self.update_intermediate_physics = True
            #     for agent in self.all_agents:
            #         agent.update_intermediate_physics = True
        agent = self.agents[self.agent_index]
        if self.update_intermediate_physics:
            # Only works for one agent!

            if agent.step_input is None:
                # First step does not call physics
                return
            while agent.possibly_partial_step() == PARTIAL_PHYSICS_STEP:
                self.render_one_frame()
            self.finish_step()
        else:
            self.render_one_frame()
            if self.render_choppy_but_realtime:
                time.sleep(1 / agent.aps)

    def render_one_frame(self):
        platform_event_loop = pyglet.app.platform_event_loop
        # pyglet_event_loop = pyglet.app.event_loop
        timeout = pyglet.app.event_loop.idle()
        platform_event_loop.step(timeout)
        if not self.render_choppy_but_realtime:
            time.sleep(self.target_dt)

    def close(self):
        if self.should_render:
            pyglet.app.is_running = False
            pyglet.app.dispatch_event('on_exit')
            pyglet.app.platform_event_loop.stop()

    def check_for_collisions(self):
        if 'DISABLE_COLLISION_CHECK' in os.environ:
            return False
        elif self.add_static_obstacle:
            for agent in self.agents:
                return check_collision_ego_obj(
                    agent.ego_rect_tuple,
                    obj2=(agent.static_obstacle_tuple,))
        elif self.is_intersection_map:
            return check_collision_agents(self.all_agents)


def main():
    env = Deepdrive2DEnv()



if __name__ == '__main__':
    if '--test_static_obstacle' in sys.argv:
        test_static_obstacle()
    else:
        main()
