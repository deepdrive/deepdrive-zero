# TODO: Multi-agent self-play
#   Combine several agents experience into the batch that PPO uses.
#   Env step will take n actions and output n observations for n agents.
#   The agents will share hyperparams,
#   Optionally height, width, and reward will be different
#   Figure out how to deal with end of episode for one actor, could just end for everybody!
import math
import os
import random
import sys
from collections import deque
import time
import json
from typing import List, Tuple
from math import pi, cos, sin
from scipy import spatial

import numpy as np
from box import Box

from deepdrive_zero.constants import VEHICLE_WIDTH, VEHICLE_HEIGHT, \
    MAX_METERS_PER_SEC_SQ, MAP_WIDTH_PX, SCREEN_MARGIN, MAP_HEIGHT_PX, \
    MAX_STEER_CHANGE_PER_SECOND, MAX_ACCEL_CHANGE_PER_SECOND, \
    MAX_BRAKE_CHANGE_PER_SECOND, STEERING_RANGE, MAX_STEER, MIN_STEER
from deepdrive_zero.constants import IS_DEBUG_MODE, GAME_OVER_PENALTY, G_ACCEL
from deepdrive_zero.experience_buffer import ExperienceBuffer
from deepdrive_zero.logs import log
from deepdrive_zero.map_gen import get_intersection
from deepdrive_zero.physics.bike_model import bike_with_friction_step, \
    get_vehicle_model
from deepdrive_zero.physics.collision_detection import get_rect, \
    get_lines_from_rect_points
from deepdrive_zero.physics.tick import physics_tick
from deepdrive_zero.utils import get_angles_ahead, get_angle, flatten_points, \
    np_rand, is_number


def get_env_config(name: str, default: float):
    env_str = os.environ.get(name.upper(), None)
    if env_str:
        name_col = f'Custom {name.lower()}'
        ret = env_str
    else:
        name_col = f'Default {name.lower()}'
        ret = default
    name_col_len = 45
    padding = ' ' * (name_col_len - len(name_col))
    log.info(f'{name_col}{padding}{ret}')
    if is_number(ret):
        return float(ret)
    elif ret.lower() in ['true', 'false']:
        return bool(ret)
    else:
        return ret

class Agent:
    def __init__(self,
                 env,
                 agent_index,
                 vehicle_width=VEHICLE_WIDTH,
                 vehicle_height=VEHICLE_HEIGHT,
                 disable_gforce_penalty=False,
                 match_angle_only=False,
                 static_map=False,
                 add_rotational_friction=True,
                 add_longitudinal_friction=True,
                 ):

        self.env = env
        self.agent_index = agent_index
        self.match_angle_only = match_angle_only
        self.static_map = static_map
        self.add_rotational_friction = add_rotational_friction
        self.add_longitudinal_friction = add_longitudinal_friction
        self.expect_normalized_actions: bool = env.expect_normalized_actions
        self.px_per_m = env.px_per_m

        # Map
        # These are duplicated per agent now as map is very small and most
        # map data is specific to agent
        self.map = None
        self.map_kd_tree = None
        self.map_flat = None

        # Static obstacle
        self.add_static_obstacle: bool = env.add_static_obstacle
        self.static_obstacle_points: np.array = None
        self.static_obst_angle_info: list = None
        self.static_obst_pixels: np.array = None
        self.static_obstacle_tuple: tuple = ()

        # Other agents
        self.other_agent_inputs: list = None

        # All units in meters and radians unless otherwise specified
        self.vehicle_width: float = vehicle_width
        self.vehicle_model:List[float] = get_vehicle_model(vehicle_width)
        self.vehicle_height: float = vehicle_height

        if 'STRAIGHT_TEST' in os.environ:
            self.num_actions = 1  # Accel
        else:
            self.num_actions: int = 3  # Steer, accel, brake
        self.prev_action: List[float] = [0] * self.num_actions
        self.prev_steer: float = 0
        self.prev_accel: float = 0
        self.prev_brake: float = 0
        self.episode_reward: float = 0
        self.speed: float = 0
        self.angle_change: float = 0
        self.fps: int = self.env.fps
        self.aps: int = self.env.aps
        self.physics_steps_per_observation = env.physics_steps_per_observation

        # Used for old waypoint per meter map
        self.map_query_seconds_ahead: np.array = np.array(
            [0.5, 1, 1.5, 2, 2.5, 3])

        if env.is_intersection_map:
            self.num_angles_ahead = 2
        elif env.is_one_waypoint_map:
            self.num_angles_ahead = 1
        else:
            self.num_angles_ahead = len(self.map_query_seconds_ahead)

        # Step properties
        self.episode_steps: int = 0
        self.num_episodes: int = 0
        self.total_steps: int = 0
        self.last_step_time: float = None
        self.wall_dt: float = None
        self.last_sleep_time: float = None

        self.total_episode_time: float = 0

        self.distance: float = None
        self.distance_to_end: float = 0
        self.prev_distance: float = None
        self.furthest_distance: float = 0
        self.velocity: np.array = np.array((0, 0))
        self.angular_velocity: float = 0

        # TODO: Use accel_magnitude internally instead so we're in SI units
        self.gforce: float = 0
        self.accel_magnitude: float = 0
        self.gforce_levels: Box = self.blank_gforce_levels()
        self.max_gforce: float = 0
        self.disable_gforce_penalty = disable_gforce_penalty
        self.prev_gforce: deque = deque(maxlen=math.ceil(env.aps))
        self.jerk: np.array = np.array((0, 0))  # m/s^3 instantaneous, i.e. frame to frame
        self.jerk_magnitude: float = 0
        self.closest_map_index: int = 0
        self.next_map_index: int = 1
        self.closest_waypoint_distance: float = 0
        self.waypoint_distances: np.array = np.array((0,0))
        self.trip_pct: float = 0
        self.avg_trip_pct: float = 0
        self._trip_pct_total: float = 0
        self.angles_ahead: List[float] = []
        self.angle_accuracies: List[float] = []
        self.episode_gforces: List[float] = []
        self.is_one_waypoint_map: bool = env.is_one_waypoint_map
        self.is_intersection_map: bool = env.is_intersection_map

        self.static_obst_angle_info: list = None

        # 0.22 m/s on 0.1
        self.max_one_waypoint_mult = 0.5  # Less than 2.5 m/s on 0.1?

        self.acceleration = np.array((0, 0))

        # Current position (center)
        self.x = None
        self.y = None

        # Angle in radians, 0 is straight up, -pi/2 is right
        self.angle = None

        self.angle_to_waypoint = None
        self.front_to_waypoint: np.array = None

        # Start position
        self.start_x = None
        self.start_y = None
        self.start_angle = None


        # Take last n (10 from 0.5 seconds) state, action, reward values and append them
        # to the observation. Should change frame rate?
        self.experience_buffer = None
        self.should_add_previous_states = '--disable-prev-states' not in sys.argv

        # TODO (research): Think about tree of neural nets for RL options

        self.ego_rect: np.array = np.array([0,0]*4)  # 4 points of ego corners
        self.ego_rect_tuple: tuple = ()  # 4 points of ego corners as tuple
        self.ego_lines: tuple = ()  # 4 edges of ego

        self.observation_space = env.observation_space
        self.collided_with: list = []

        self.done: bool = False

        self.prev_desired_accel = 0
        self.prev_desired_steer = 0
        self.prev_desired_brake = 0

        log.info(f'Agent {self.agent_index} config '
                 f'--------------------')
        self.jerk_penalty_coeff = \
            get_env_config('JERK_PENALTY_COEFF', default=0.10)
        self.gforce_penalty_coeff = \
            get_env_config('GFORCE_PENALTY_COEFF', default=0.031)
        self.lane_penalty_coeff = \
            get_env_config('LANE_PENALTY_COEFF', default=0.02)
        self.collision_penalty_coeff = \
            get_env_config('COLLISION_PENALTY_COEFF', default=0.31)
        self.speed_reward_coeff = \
            get_env_config('SPEED_REWARD_COEFF', default=0.50)
        self.win_coefficient = \
            get_env_config('WIN_COEFF', default=1)
        self.end_on_harmful_gs = \
            bool(get_env_config('END_ON_HARMFUL_GS', default=True))
        self.constrain_controls = \
            bool(get_env_config('CONSTRAIN_CONTROLS', default=True))
        self.ignore_brake = \
            bool(get_env_config('IGNORE_BRAKE', default=False))
        self.forbid_deceleration = \
            bool(get_env_config('FORBID_DECELERATION',
                                default=env.forbid_deceleration))
        self.expect_normalized_action_deltas = \
            bool(get_env_config('EXPECT_NORMALIZED_ACTION_DELTAS',
                                default=env.expect_normalized_action_deltas))
        self.incent_win = bool(get_env_config('INCENT_WIN',
                                              default=env.incent_win))


        self.max_steer_change_per_tick = MAX_STEER_CHANGE_PER_SECOND / self.fps
        self.max_accel_change_per_tick = MAX_ACCEL_CHANGE_PER_SECOND / self.fps
        self.max_brake_change_per_tick = MAX_BRAKE_CHANGE_PER_SECOND / self.fps
        self.max_steer_change = self.max_steer_change_per_tick * self.physics_steps_per_observation
        self.max_accel_change = self.max_accel_change_per_tick * self.physics_steps_per_observation
        self.max_brake_change = self.max_brake_change_per_tick * self.physics_steps_per_observation

        self.max_accel_historical = -np.inf

        self.reset()

    @log.catch
    def step(self, action):
        dt = self.env.get_dt()
        info = Box(default_box=True)
        if 'STRAIGHT_TEST' in os.environ:
            accel = action[0]
            steer = 0
            brake = 0
        else:
            steer, accel, brake = action

        # if accel > self.max_accel_historical:
        #     log.info(f'new max accel {accel}')
        #     self.max_accel_historical = accel

        steer, accel, brake = self.denormalize_actions(steer, accel, brake)
        self.prev_desired_steer = steer
        self.prev_desired_accel = accel
        self.prev_desired_brake = brake

        if 'STRAIGHT_TEST' in os.environ:
            steer = 0
            brake = 0
            # accel = MAX_METERS_PER_SEC_SQ

        if '--simple-steer' in sys.argv and self.angles_ahead:
            accel = MAX_METERS_PER_SEC_SQ * 0.7
            steer = -self.angles_ahead[0]
        elif self.match_angle_only:
            accel = MAX_METERS_PER_SEC_SQ * 0.7

        info.stats.steer = steer
        info.stats.accel = accel
        info.stats.brake = brake
        info.stats.speed = self.speed
        info.stats.episode_time = self.env.total_episode_time

        now = time.time()
        if self.last_step_time is None:
            # init
            self.last_step_time = now
            reward = 0
            done = False
            observation = self.get_blank_observation()
        else:
            collided = bool(self.collided_with)

            obs_data = self.get_observation(steer, accel, brake, dt, info)
            (lane_deviation, observation, closest_map_point,
             left_lane_distance, right_lane_distance) = obs_data

            done, won, lost = self.get_done(closest_map_point, lane_deviation,
                                            collided)
            reward, info = self.get_reward(
                lane_deviation, won, lost, collided, info, steer, accel,
                left_lane_distance, right_lane_distance
            )
            info.stats.lane_deviation = lane_deviation
            step_time = now - self.last_step_time
            # log.trace(f'step time {round(step_time, 3)}')

            if done:
                info.stats.all_time.won = won

        self.last_step_time = now
        self.episode_reward += reward

        self.prev_action = action
        self.episode_steps += 1

        if done:
            self.num_episodes += 1
            self._trip_pct_total += self.trip_pct
            self.avg_trip_pct = self._trip_pct_total / self.num_episodes
            episode_angle_accuracy = np.array(self.angle_accuracies).mean()
            episode_gforce_avg = np.array(self.episode_gforces).mean()
            log.debug(f'Score {round(self.episode_reward, 2)}, '
                      f'Rew/Step: {self.episode_reward/self.episode_steps}, '
                      f'Steps: {self.episode_steps}, '
                      # f'Closest map indx: {self.closest_map_index}, '
                      f'Distance {round(self.distance, 2)}, '
                      f'Wp {self.next_map_index - 1}, '
                      f'Angular velocity {round(self.angular_velocity, 2)}, '
                      f'Speed: {round(self.speed, 2)}, '
                      f'Max gforce: {round(self.max_gforce, 4)}, '
                      f'Avg gforce: {round(episode_gforce_avg, 4)}, '
                      f'Trip pct {round(self.trip_pct, 2)}, '
                      f'Angle accuracy {round(episode_angle_accuracy, 2)}, '
                      f'Agent index {round(self.agent_index, 2)}, '
                      f'Total steps {self.total_steps}, '
                      f'Env ep# {self.env.num_episodes}, '
                      f'Ep# {self.num_episodes}')

        self.total_steps += 1

        self.set_calculated_props()

        return observation, reward, done, info.to_dict()

    # TODO: Set these properties only when x,y, or angle change
    @property
    def front_x(self):
        """Front middle x position of ego"""
        theta = pi / 2 + self.angle
        return self.x + cos(theta) * self.vehicle_height / 2

    @property
    def front_y(self):
        """Front middle y position of ego"""
        theta = pi / 2 + self.angle
        return self.y + sin(theta) * self.vehicle_height / 2

    @property
    def back_x(self):
        """Front middle x position of ego"""
        theta = pi / 2 + self.angle
        return self.x - cos(theta) * self.vehicle_height / 2

    @property
    def back_y(self):
        """Front middle y position of ego"""
        theta = pi / 2 + self.angle
        return self.y - sin(theta) * self.vehicle_height / 2

    @property
    def front_pos(self):
        return np.array((self.front_x, self.front_y))

    @property
    def ego_pos(self):
        return np.array((self.x, self.y))

    @property
    def heading(self):
        return np.array([self.front_x, self.front_y]) - self.ego_pos


    def denormalize_actions(self, steer, accel, brake):

        if self.expect_normalized_action_deltas:
            steer, accel, brake = self.check_action_bounds(accel, brake, steer)
            steer *= self.max_steer_change
            accel *= self.max_accel_change
            if self.forbid_deceleration:
                accel = self.max_accel_change * ((1 + accel) / 2)  # Positive only
            else:
                accel = accel * self.max_accel_change
            brake = self.max_brake_change * ((1 + brake) / 2)  # Positive only

            steer += self.prev_steer
            accel += self.prev_accel
            brake += self.prev_brake

            steer = min(steer, MAX_STEER)
            steer = max(steer, MIN_STEER)

            accel = min(accel, MAX_METERS_PER_SEC_SQ)
            accel = max(accel, -MAX_METERS_PER_SEC_SQ)  # TODO: Lower max reverse accel and speed

            brake = min(brake, MAX_METERS_PER_SEC_SQ)
            brake = max(brake, -MAX_METERS_PER_SEC_SQ)

        elif self.expect_normalized_actions:
            steer, accel, brake = self.check_action_bounds(accel, brake, steer)
            steer = steer * STEERING_RANGE
            if self.forbid_deceleration:
                accel = MAX_METERS_PER_SEC_SQ * ((1 + accel) / 2)  # Positive only
            else:
                accel *= MAX_METERS_PER_SEC_SQ
            brake = MAX_METERS_PER_SEC_SQ * ((1 + brake) / 2)  # Positive only
        return steer, accel, brake

    # TODO: Numba this (return string or enum in order to log)
    def check_action_bounds(self, accel, brake, steer):
        if not (-1 <= accel <= 1):
            log.trace(f'Found accel outside -1=>1 of {accel}')
            accel = max(-1, accel)
            accel = min(1, accel)
        if not (-1 <= steer <= 1):
            log.trace(f'Found steer outside -1=>1 of {steer}')
            steer = max(-1, steer)
            steer = min(1, steer)
        if not (-1 <= brake <= 1):
            log.trace(f'Found steer outside -1=>1 of {brake}')
            brake = max(-1, brake)
            brake = min(1, brake)
        return steer, accel, brake

    @staticmethod
    def blank_gforce_levels():
        return Box(harmful=False, jarring=False,
                   uncomfortable=False)

    def get_blank_observation(self):
        ret = self.populate_observation(
            steer=0,
            brake=0,
            accel=0,
            closest_map_point=self.map.waypoints[0],
            lane_deviation=0,
            angles_ahead=[0] * self.num_angles_ahead,
            harmful_gs=False,
            jarring_gs=False,
            uncomfortable_gs=False,
            is_blank=True,)
        return ret

    def populate_observation(self, closest_map_point, lane_deviation,
                             angles_ahead, steer, brake, accel, harmful_gs,
                             jarring_gs, uncomfortable_gs,
                             left_lane_distance=0,
                             right_lane_distance=0, is_blank=False):
        observation = \
            Box(steer=steer,
                accel=accel,
                brake=brake,
                x=self.x,
                y=self.y,
                angle=self.angle,
                speed=self.speed,
                harmful_gs=float(harmful_gs),
                jarring_gs=float(jarring_gs),
                uncomfortable_gs=float(uncomfortable_gs),
                distance=self.distance,
                angle_change=self.angle_change,
                closest_map_point_x=closest_map_point[0],
                closest_map_point_y=closest_map_point[1],
                lane_deviation=lane_deviation,
                vehicle_width=self.vehicle_width,
                vehicle_height=self.vehicle_height,
                cog_to_front_axle=self.vehicle_model[0],
                cog_to_rear_axle=self.vehicle_model[1],
                left_lane_distance=left_lane_distance,
                right_lane_distance=right_lane_distance)

        done_input = 1 if self.done else 0
        if self.is_intersection_map:
            other_agent_inputs = self.get_other_agent_inputs(is_blank)
            if len(angles_ahead) == 1:
                intersection_angles_ahead = [angles_ahead[0], angles_ahead[0]]
            else:
                intersection_angles_ahead = angles_ahead
            # if self.agent_index == 0:
                # log.debug(
                #     f'angles ahead {intersection_angles_ahead}\n'
                    # f'prev_steer {self.prev_steer}\n'
                    # f'prev_accel {self.prev_accel}\n'
                    # f'speed {self.speed}\n'
                    # f'left {left_lane_distance}\n'
                    # f'right {right_lane_distance}\n'
                    # f'done {done_input}\n'
                    # f'waypoint_distances {self.waypoint_distances}\n'
                    # f'velocity {self.velocity}\n'
                    # f'acceleration {self.acceleration}\n'
                    # f'agents {other_agent_inputs}\n'
                # )

        if self.env.return_observation_as_array:
            # TODO: These model inputs should be recorded somehow so we can
            #   use the trained models later on without needing this code.

            # TODO: Normalize these and ensure they don't exceed reasonable
            #   physical bounds
            common_inputs = [
                # Previous outputs (TODO: Remove for recurrent models like r2d1 / lstm / gtrxl?)
                self.prev_desired_steer,
                self.prev_desired_accel,
                self.prev_desired_brake,

                # Previous outputs after physical constraints applied
                self.prev_steer,
                self.prev_accel,
                self.prev_brake,

                self.speed,
                self.accel_magnitude,
                self.jerk_magnitude,
                # why? done_input,
                # self.env.target_dt,  # TODO: Add dt noise from domain randomization as input
                # Normalize these before adding: self.distance_to_end,
                # Normalize these before adding: self.distance,
            ]
            if self.is_one_waypoint_map:
                if self.match_angle_only:
                    return np.array([angles_ahead[0], self.prev_steer])
                elif 'STRAIGHT_TEST' in os.environ:
                    return np.array(common_inputs + [self.speed])
                else:
                    ret = [angles_ahead[0], self.prev_steer, self.prev_accel,
                           self.speed, self.distance_to_end]
                    if self.env.add_static_obstacle:
                        ret += self.get_static_obstacle_inputs(is_blank)
                    if is_blank:
                        return np.array(ret) * 0
                    else:
                        return np.array(ret)
            elif self.is_intersection_map:
                # TODO: Move get_intersection_observation here
                ret = common_inputs + \
                      [intersection_angles_ahead[0],
                       intersection_angles_ahead[1],
                       left_lane_distance,
                       right_lane_distance]
                if is_blank:
                    self.set_distance()
                ret += list(self.waypoint_distances)
                ret += list(self.velocity)
                ret += list(self.acceleration)

                ret += other_agent_inputs

                if is_blank:
                    return np.array(ret) * 0
                else:
                    return np.array(ret)
            else:
                observation = np.array(observation.values())
                observation = np.concatenate((observation, angles_ahead),
                                             axis=None)

                if self.should_add_previous_states:
                    if self.experience_buffer.size() == 0:
                        self.experience_buffer.setup(shape=(len(observation),))
                    if is_blank:
                        past_values = np.concatenate(np.array(
                            self.experience_buffer.blank_buffer), axis=0)
                    else:
                        self.experience_buffer.maybe_add(
                            observation, self.env.total_episode_time)
                        past_values = np.concatenate(np.array(
                            self.experience_buffer.buffer), axis=0)
                    observation = np.concatenate((observation, past_values),
                                                 axis=None)
                if math.nan in observation:
                    raise RuntimeError(f'Found NaN in observation')
                if np.nan in observation:
                    raise RuntimeError(f'Found NaN in observation')
                if -math.inf in observation or math.inf in observation:
                    raise RuntimeError(f'Found inf in observation')

        else:
            observation.angles_ahead = angles_ahead
            if self.env.add_static_obstacle:
                self.get_static_obstacle_inputs()

        return observation

    def get_other_agent_inputs(self, is_blank=False):

        # TODO: Perhaps we should feed this into a transformer / LSTM / or
        #  use attention as the number of agents can be variable in length and
        #  may exceed the amount of input we want to pass to the net.
        ret = []
        v = self.velocity
        ang = self.get_angle_to_point
        dst = np.linalg.norm
        f = self.front_pos

        # TODO: These should be sorted by time to collision TTC where
        #   TTC is approximated by assuming vehicles immediately change
        #   direction towards each other at current velocity.

        for i in range(self.env.num_agents):
            if i == self.agent_index:
                continue
            if is_blank:
                agent = self
            else:
                agent = self.env.agents[i]
            ret += list(agent.velocity - v)  # ego relative velocity
            ret += list(agent.velocity)
            ret += list(agent.acceleration)
            for p in agent.ego_rect:
                ret.append(ang(p))
                ret.append(dst(p - f))

        if is_blank:
            ret = list(np.array(ret) * 0)
        self.other_agent_inputs = ret
        return ret

    def get_static_obstacle_inputs(self, is_blank=False):
        start_static_obs = self.static_obstacle_points[0]
        end_static_obs = self.static_obstacle_points[1]
        start_obst_angle = self.get_angle_to_point(start_static_obs)
        end_obst_angle = self.get_angle_to_point(end_static_obs)
        start_obst_dist = np.linalg.norm(start_static_obs - self.front_pos)
        end_obst_dist = np.linalg.norm(end_static_obs - self.front_pos)
        ret = [start_obst_dist, end_obst_dist, start_obst_angle,
               end_obst_angle]
        self.static_obst_angle_info = ret

        # log.info(f'start obs angle {math.degrees(start_obst_angle)}')
        if is_blank:
            ret = list(np.array(ret) * 0)

        return ret

    def reset(self):
        self.angle = self.start_angle
        self.angle_change = 0
        self.x = self.start_x
        self.y = self.start_y
        self.angle_change = 0
        self.speed = 0
        self.episode_reward = 0
        self.episode_steps = 0
        self.distance = None
        self.prev_distance = None
        self.furthest_distance = 0
        self.velocity = np.array((0,0))
        self.angular_velocity = 0
        self.acceleration = np.array((0,0))
        self.gforce = 0
        self.accel_magnitude = 0
        self.jerk = 1
        self.jerk_magnitude = 0
        self.gforce_levels = self.blank_gforce_levels()
        self.max_gforce = 0
        self.closest_map_index = 0
        self.next_map_index = 1
        self.trip_pct = 0
        self.angles_ahead = []
        self.angle_accuracies = []
        self.episode_gforces = []
        self.collided_with = []
        self.done = False
        self.prev_accel = 0
        self.prev_steer = 0
        self.prev_brake = 0

        # TODO: Regen map every so often
        if self.map is None or not self.static_map:
            self.gen_map()

        self.set_calculated_props()

        if self.experience_buffer is None:
            self.experience_buffer = ExperienceBuffer()
        self.experience_buffer.reset()
        obz = self.get_blank_observation()

        return obz

    def set_calculated_props(self):
        self.ego_rect, self.ego_rect_tuple = get_rect(
            self.x, self.y, self.angle, self.vehicle_width, self.vehicle_height)

        self.ego_lines = get_lines_from_rect_points(self.ego_rect_tuple)

    def get_done(self, closest_map_point, lane_deviation,
                 collided: bool) -> Tuple[bool, bool, bool]:
        done = False
        won = False
        lost = False
        if 'DISABLE_GAME_OVER' in os.environ:
            return done, won, lost
        elif collided:
            log.warning(f'Collision, game over agent {self.agent_index}')
            done = True
            lost = True
        elif self.gforce > 1.0 and self.end_on_harmful_gs:
            # Only end on g-force once we've learned to complete part of the trip.
            log.warning(f'Harmful g-forces, game over agent {self.agent_index}')
            done = True
            lost = True
        elif (self.episode_steps + 1) % self.env._max_episode_steps == 0:
            log.warning(f"Time's up agent {self.agent_index}")
            done = True
        elif 'DISABLE_CIRCLE_CHECK' not in os.environ and \
                abs(math.degrees(self.angle)) > 400:
            done = True
            lost = True
            log.warning(f'Going in circles - angle {math.degrees(self.angle)} too high')
        elif self.is_one_waypoint_map or self.is_intersection_map:
            if self.is_intersection_map and \
                    self.closest_map_index > self.next_map_index:
                # Negative progress catches this first depending on distance
                # thresholds
                done = True
                lost = True
                log.warning(f'Skipped waypoint {self.next_map_index} '
                            f'agent {self.agent_index}')
            elif (self.furthest_distance - self.distance) > 2:
                done = True
                lost = True
                log.warning(f'Negative progress agent {self.agent_index}')
            elif abs(self.map.length - self.distance) < 1:
                done = True
                won = True
                # You win!
                log.success(f'Reached destination! '
                            f'Steps: {self.episode_steps} '
                            f'Agent: {self.agent_index}')
        elif list(self.map.waypoints[-1]) == list(closest_map_point):
            # You win!
            done = True
            won = True
            log.success(f'Reached destination! '
                        f'Steps: {self.episode_steps}')
        if '--test-win' in sys.argv:
            won = True
        self.done = done
        return done, won, lost


    def get_reward(self, lane_deviation: float,  won: bool, lost: bool,
                   collided: bool, info: Box, steer: float,
                   accel: float, left_lane_distance: float,
                   right_lane_distance: float) -> Tuple[float, Box]:

        angle_diff = abs(self.angles_ahead[0])

        if 'STRAIGHT_TEST' in os.environ:
            angle_reward = 0
        else:
            angle_reward = 4 * pi - angle_diff

        angle_accuracy = 1 - angle_diff / (2 * pi)

        # TODO: Fix incentive to drive towards waypoint instead of edge of
        #   static obstacle. Perhaps just remove this reward altogether
        #   in favor of end of episode reward.
        frame_distance = self.distance - self.prev_distance
        speed_reward = frame_distance * self.speed_reward_coeff

        # TODO: Idea penalize residuals of a quadratic regression fit to history
        #  of actions. Currently penalizing jerk instead which may or may not
        #  be better (but it is simpler).
        if 'ACTION_PENALTY' in os.environ:
            action_penalty = float(os.environ['ACTION_PENALTY'])
            steer_penalty = abs(self.prev_steer - steer) * action_penalty
            accel_penalty = abs(self.prev_accel - accel) * action_penalty
        else:
            steer_penalty = 0
            accel_penalty = 0


        gforce_penalty = 0
        if not self.disable_gforce_penalty and self.gforce > 0.05:
            # Note that it's useful to give a low or no g-force penalty
            # at first, then to scale it up once waypoints are being reached.
            gforce_penalty = self.gforce_penalty_coeff * self.gforce

        jerk_magnitude = np.linalg.norm(self.jerk)
        info.stats.jerk = jerk_magnitude
        jerk_penalty = self.jerk_penalty_coeff * jerk_magnitude
        self.jerk_magnitude = jerk_magnitude

        lane_penalty = 0
        if left_lane_distance < 0:
            lane_penalty += abs(left_lane_distance)
        if right_lane_distance < 0:
            # yes both can happen if you're orthogonal to the lane
            lane_penalty += abs(right_lane_distance)
        lane_penalty *= self.lane_penalty_coeff

        # if self.agent_index == 1:
        #     log.info(f'left distance {left_lane_distance} '
        #              f'right distance {right_lane_distance} '
        #              f'agent {self.agent_index}')

        self.accel_magnitude = self.gforce * G_ACCEL

        # log.debug(f'jerk {round(jerk_magnitude)} accel {round(accel_magnitude)}')

        self.angle_accuracies.append(angle_accuracy)
        info.stats.angle_accuracy = angle_accuracy

        if collided:
            # TODO: Make dependent on |Δv|
            collision_penalty = self.collision_penalty_coeff
        else:
            collision_penalty = 0

        win_reward = self.get_win_reward(won)
        ret = (
           + speed_reward
           + win_reward
           - gforce_penalty
           - collision_penalty
           - jerk_penalty
           - lane_penalty
        )

        # IDEA: Induce curriculum by zeroing things like static obstacle
        # until we've learned to steer smoothly. Alternatively, we could
        # train with the full complexity, then fine-tune to improve
        # smoothness.

        # if self.agent_index == 0:
        #     log.debug(
        #         f'reward {ret} '
        #         f'next waypoint {self.next_map_index} '
        #         f'distance {self.distance} '
        #         f'speed {speed_reward} '
        #         f'gforce {gforce_penalty} '
        #         f'jerk {jerk_penalty} '
        #         f'lane {lane_penalty} '
        #         f'win {win_reward} '
        #         f'collision {collision_penalty} '
        #         f'steer {steer_penalty} '
        #         f'accel {accel_penalty} '
        #     )

        return ret, info

    def get_win_reward(self, won):
        win_reward = 0
        if self.incent_win and won:
            win_reward = self.win_coefficient
        return win_reward

    def get_observation(self, steer, accel, brake, dt, info):
        if self.ignore_brake:
            brake = False
        if self.speed > 100:
            log.warning('Cutting off throttle at speed > 100m/s')
            accel = 0

        self.step_physics(dt, steer, accel, brake, info)

        closest_map_point, closest_map_index, closest_waypoint_distance = \
            get_closest_point((self.front_x, self.front_y), self.map_kd_tree)

        self.closest_waypoint_distance = closest_waypoint_distance
        self.closest_map_index = closest_map_index
        self.set_distance()

        self.trip_pct = 100 * self.distance / self.map.length

        half_lane_width = self.map.lane_width / 2
        left_lane_distance = right_lane_distance = half_lane_width

        if self.is_one_waypoint_map:
            angles_ahead = self.get_one_waypoint_angle_ahead()

            # log.info(f'angle ahead {math.degrees(angles_ahead[0])}')
            # log.info(f'angle {math.degrees(self.angle)}')
        elif self.is_intersection_map:
            angles_ahead, left_lane_distance, right_lane_distance, = \
                self.get_intersection_observation(half_lane_width,
                                                  left_lane_distance,
                                                  right_lane_distance)
        else:
            self.trip_pct = 100 * closest_map_index / (len(self.map.waypoints) - 1)
            angles_ahead = self.get_angles_ahead(closest_map_index)

        # log.trace(f'lane dist left {left_lane_distance} '
        #          f'right {right_lane_distance}')

        self.angles_ahead = angles_ahead

        info.stats.closest_map_index = closest_map_index
        info.stats.trip_pct = self.trip_pct
        info.stats.distance = self.distance

        observation = self.populate_observation(
            closest_map_point=closest_map_point,
            lane_deviation=closest_waypoint_distance,
            angles_ahead=angles_ahead,
            steer=steer,
            brake=brake,
            accel=accel,
            harmful_gs=self.gforce_levels.harmful,
            jarring_gs=self.gforce_levels.jarring,
            uncomfortable_gs=self.gforce_levels.uncomfortable,
            left_lane_distance=left_lane_distance,
            right_lane_distance=right_lane_distance,
        )

        return (closest_waypoint_distance, observation, closest_map_point,
                left_lane_distance, right_lane_distance)


    # TOOD: Numba this
    def get_intersection_observation(self, half_lane_width, left_distance,
                                     right_distance):
        # TODO: Move other agent observations (like distances) here
        a2w = self.get_angle_to_point
        wi = self.next_map_index
        mp = self.map
        angles_ahead = [a2w(p) for p in mp.waypoints[wi:wi+2]]
        if self.agent_index == 0:
            # left turn agent
            if self.front_y < mp.waypoints[1][1]:
                # Before entering intersection
                x = mp.waypoints[0][0]
                left_x = x - half_lane_width
                right_x = x + half_lane_width
                left_distance = min(self.ego_rect.T[0]) - left_x
                right_distance = right_x - max(self.ego_rect.T[0])
            else:
                if self.back_x < mp.waypoints[2][0]:
                    # After exiting intersection
                    y = mp.waypoints[2][1]
                    bottom_y = y - half_lane_width
                    top_y = y + half_lane_width
                    left_distance = min(self.ego_rect.T[1]) - bottom_y
                    right_distance = top_y - max(self.ego_rect.T[1])
        else:
            # Straight agent
            x = mp.waypoints[0][0]
            left_x = x - half_lane_width
            right_x = x + half_lane_width
            right_distance = min(self.ego_rect.T[0]) - left_x
            left_distance = right_x - max(self.ego_rect.T[0])

        return angles_ahead, left_distance, right_distance

    def get_angles_ahead(self, closest_map_index):
        """
        Note: this assumes we are on the old meter per waypoint map path.
        This meter per waypoint map is not being used for current training
        but we may want to bring it back as we are able to achieve more complex
        behavior.
        """
        distances = self.map.distances

        return get_angles_ahead(total_points=len(distances),
                                total_length=self.map.length,
                                speed=self.speed,
                                map_points=self.map.waypoints,
                                ego_angle=self.angle,
                                seconds_ahead=self.map_query_seconds_ahead,
                                closest_map_index=closest_map_index,
                                heading=self.heading,
                                ego_front=self.ego_pos)

    def get_one_waypoint_angle_ahead(self):
        waypoint = np.array((self.map.x[-1], self.map.y[-1]))
        return self.get_angle_to_waypoint(waypoint)

    def get_angle_to_waypoint(self, waypoint):
        angle_to_waypoint = self.get_angle_to_point(waypoint)
        self.angle_to_waypoint = angle_to_waypoint
        front_to_waypoint_vector = waypoint - self.front_pos
        self.front_to_waypoint = front_to_waypoint_vector

        # Repeat to match dimensions of many waypoint map
        ret = [angle_to_waypoint] * self.num_angles_ahead

        return ret

    def get_angle_to_point(self, p):
        """
        Computes steering angle required for ego to change heading towards `p`
        :param p:
        :return:
        """
        front_to_point_vector = p - self.front_pos
        ret = get_angle(self.heading, front_to_point_vector)
        return ret


    def set_distance(self):
        mp = self.map
        if self.closest_map_index == self.next_map_index and \
                self.closest_waypoint_distance < 1 and \
                self.next_map_index < len(self.map.x) - 1:
            # Advance to next waypoint
            self.next_map_index += 1

        self.prev_distance = self.distance
        if 'STRAIGHT_TEST' in os.environ:
            self.distance = self.x - self.start_x
        elif self.is_one_waypoint_map:
            end = np.array([mp.x[-1], mp.y[-1]])

            # TODO: Use self.front_pos
            pos = np.array([self.front_x, self.front_y])
            self.distance_to_end = np.linalg.norm(end - pos)

            self.distance = mp.length - self.distance_to_end
        elif self.is_intersection_map and self.env.agents is not None:
            # Add the next waypoint distance to the beginning of the array,
            # then add the remaining distances.
            # Finally pad with zeroes at the end to keep a static length.
            # So with 3 waypoints distances would take on the following schema,
            # where waypoint 0 is the start point.
            # [wp1dist, wp2dist] => before reaching first waypoint
            # [wp2dist, 0] => after waypoint 1, before waypoint 2
            # Also, for multi-agent, we need the max number of waypoints for
            # all agents which is why self.env.agents must be set
            max_waypoints = max(len(a.map.waypoints) for a in self.env.agents)  # TODO: Avoid recalculating this - do it on last agent map gen or create env map gen
            waypoint_distances = np.zeros((max_waypoints - 1,))
            next_index = self.next_map_index
            for i in range(len(self.map.waypoints) - next_index):
                wi = next_index + i
                next_pos = np.array([mp.x[wi], mp.y[wi]])
                dist = np.linalg.norm(next_pos - self.front_pos)
                waypoint_distances[i] = dist
            self.distance = (mp.distances[self.next_map_index] -
                             abs(waypoint_distances[0]))
            self.waypoint_distances = waypoint_distances
            # log.debug(waypoint_distances)
        else:
            # Assumes waypoints are very close, i.e. 1m apart
            self.distance = mp.distances[self.closest_map_index]
        # log.debug(f'distance {self.distance}')
        self.furthest_distance = max(self.distance, self.furthest_distance)
        if self.prev_distance is None:
            # Init prev distance
            self.prev_distance = self.distance

    def gen_map(self):
        # TODO: Move map to env.py. Right now the map is really just a couple
        #   of waypoints, so it's fine to duplicate it for each agent.

        lane_width = 10 * 0.3048
        # Generate one waypoint map
        if self.is_one_waypoint_map:
            x_norm, y_norm = self.gen_one_waypoint_map()
            x_pixels = x_norm * MAP_WIDTH_PX + SCREEN_MARGIN
            y_pixels = y_norm * MAP_HEIGHT_PX + SCREEN_MARGIN
            x_meters = x_pixels / self.px_per_m
            y_meters = y_pixels / self.px_per_m
        elif self.is_intersection_map:
            x_meters, y_meters, lane_width, lane_lines = \
                self.gen_intersection_map()
            x_pixels = x_meters * self.px_per_m
            y_pixels = y_meters * self.px_per_m
        else:
            raise NotImplementedError()

        waypoints = list(zip(list(x_meters), list(y_meters)))

        distances = np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
        distances = np.concatenate((np.array([0]), distances))

        self.map = Box(x=x_meters,
                       y=y_meters,
                       x_pixels=x_pixels,
                       y_pixels=y_pixels,
                       waypoints=waypoints,
                       distances=distances,
                       length=distances[-1],
                       width=(MAP_WIDTH_PX + SCREEN_MARGIN) / self.px_per_m,
                       height=(MAP_HEIGHT_PX + SCREEN_MARGIN) / self.px_per_m,
                       static_obst_pixels=self.static_obst_pixels,
                       lane_width=lane_width)

        self.x = self.map.x[0]
        self.y = self.map.y[0]

        self.start_x = self.x
        self.start_y = self.y

        # Physics properties
        # x is right, y is straight
        self.map_kd_tree = spatial.KDTree(self.map.waypoints)

        self.map_flat = flatten_points(self.map.waypoints)
        if self.is_one_waypoint_map:
            self.angle = -pi / 2
        elif self.is_intersection_map:
            if self.agent_index == 0:
                self.angle = 0
            elif self.agent_index == 1:
                self.angle = pi
            else:
                raise NotImplementedError('More than 2 agents not supported')

        else:
            raise NotImplementedError()
            # self.angle = self.get_start_angle()

        self.start_angle = self.angle

    def gen_one_waypoint_map(self):
        m = self.max_one_waypoint_mult
        x1 = 0.1
        y1 = 0.5
        if self.static_map:
            x2 = x1 + 0.2 * m + 0.1
            y2 = y1 + (2 * 0.2 - 1) * m
        else:
            x2 = x1 + np_rand() * m + 0.1
            y2 = y1 + (2 * np_rand() - 1) * m
        x = np.array([x1, x2])
        y = np.array([y1, y2])
        if self.add_static_obstacle:
            _, self.static_obst_pixels = \
                get_static_obst(m, x, y)

            # Need to work backward from pixels to incorporate
            # screen margin offset
            self.static_obstacle_points = \
                self.static_obst_pixels / self.px_per_m

            self.static_obstacle_tuple = tuple(
                map(tuple, self.static_obstacle_points.tolist()))
        return x, y

    def gen_intersection_map(self):
        lines, lane_width = get_intersection()
        left_vert, mid_vert, right_vert, top_horiz, mid_horiz, bottom_horiz = \
            lines

        # Get waypoints
        wps = []
        if self.agent_index == 0:
            wps.append((27.0770290995851, random.uniform(6, 18)))
            wps.append((mid_vert[0][0] + lane_width / 2, bottom_horiz[0][1]))
            wps.append((left_vert[0][0], mid_horiz[0][1] + lane_width / 2))
            wps.append((1.840549443086846, mid_horiz[0][1] + lane_width / 2))
        elif self.agent_index == 1:
            wps.append((mid_vert[0][0] - lane_width / 2, random.uniform(33, 47)))
            # wps.append((mid_vert[0][0] - lane_width / 2, 40.139197872452702))
            wps.append((mid_vert[0][0] - lane_width / 2, 4.139197872452702))
        else:
            raise NotImplementedError('More than 2 agents not yet supported')

        x, y = np.array(list(zip(*wps)))

        return x, y, lane_width, lines

    def step_physics(self, dt, steer, accel, brake, info):
        n = self.env.physics_steps_per_observation
        start = time.time()
        (self.acceleration, self.angle, self.angle_change,
         self.angular_velocity, self.gforce, self.jerk, self.max_gforce,
         self.speed, self.x, self.y, self.prev_accel, self.prev_brake,
         self.prev_steer, self.velocity) = physics_tick(
            accel, self.add_longitudinal_friction, self.add_rotational_friction,
            brake, self.acceleration, self.angle, self.angle_change,
            self.angular_velocity, self.gforce, self.max_gforce,
            self.speed, self.velocity, self.x, self.y, dt, n, self.prev_accel,
            self.prev_brake, self.prev_steer, steer, self.vehicle_model,
            self.ignore_brake, self.constrain_controls, self.max_steer_change_per_tick,
            self.max_accel_change_per_tick)
        # log.debug(f'step took {time.time() - start}s')

        info.stats.gforce = self.gforce
        self.total_episode_time += dt * n
        self.env.total_episode_time += dt * n

        self.ego_rect, self.ego_rect_tuple = get_rect(
            self.x, self.y, self.angle, self.vehicle_width, self.vehicle_height)

        self.episode_gforces.append(self.gforce)


def get_closest_point(point, kd_tree):
    distance, index = kd_tree.query(point)
    if index >= kd_tree.n:
        log.warning(f'kd tree index out of range, using last index. '
                    f'point {point}\n'
                    f'kd_tree: {json.dumps(kd_tree.data.tolist(), indent=2)}')
        index = kd_tree.n - 1
    point = kd_tree.data[index]
    return point, index, distance


def get_static_obst(m, x, y):
    # Get point between here + 2 car lengths and destination
    # Draw random size / angle line
    # TODO: Allow circular obstacles using equation of circle and
    #  set equal to equation for line
    # xy = np.dstack((x, y))
    # xy_dist = np.diff(xy, axis=1)[0][0]
    # rand_vals = np.random.rand(2)
    x_dist = x[1] - x[0]
    y_dist = y[1] - y[0]

    total_dist = np.linalg.norm([x_dist, y_dist])
    center_dist = np.random.rand() * total_dist * 0.6 + 0.1
    theta = np.arctan(y_dist / x_dist)
    obst_center_x = cos(theta) * center_dist + x[0]
    obst_center_y = sin(theta) * center_dist + y[0]

    obst_angle = np.random.rand() * pi
    obst_width = np.random.rand() * 0.1 + 0.025
    obst_end_x = obst_center_x + cos(obst_angle) * obst_width / 2
    obst_end_y = obst_center_y + sin(obst_angle) * obst_width / 2
    obst_beg_x = obst_center_x - cos(obst_angle) * obst_width / 2
    obst_beg_y = obst_center_y - sin(obst_angle) * obst_width / 2
    static_obst_x = np.array([obst_beg_x, obst_end_x])
    static_obst_y = np.array([obst_beg_y, obst_end_y])
    static_obst = np.dstack(
        (static_obst_x, static_obst_y))[0]
    static_obst_x_pixels = static_obst_x * MAP_WIDTH_PX + SCREEN_MARGIN
    static_obst_y_pixels = static_obst_y * MAP_HEIGHT_PX + SCREEN_MARGIN
    static_obst_pixels = np.dstack(
        (static_obst_x_pixels, static_obst_y_pixels))[0]
    return static_obst, static_obst_pixels


def test_static_obstacle():
    points, pixels = get_static_obst(None, np.array([0, 1]), np.array([0, 1]))