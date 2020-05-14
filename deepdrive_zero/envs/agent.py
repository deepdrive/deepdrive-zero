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

from deepdrive_zero.constants import G_ACCEL, \
    VEHICLE_WIDTH, VEHICLE_LENGTH, MAX_STEER_CHANGE_PER_SECOND, \
    MAX_ACCEL_CHANGE_PER_SECOND, MAX_BRAKE_CHANGE_PER_SECOND, STEERING_RANGE, \
    MAX_METERS_PER_SEC_SQ, MAX_BRAKE_G, PARTIAL_PHYSICS_STEP, MAX_STEER, \
    MIN_STEER, RIGHT_HAND_TRAFFIC, MAP_WIDTH_PX, SCREEN_MARGIN, MAP_HEIGHT_PX
from deepdrive_zero.discrete.comfortable_actions import COMFORTABLE_ACTIONS, \
    COMFORTABLE_ACTIONS_IDLE, COMFORTABLE_ACTIONS_DECAY_STEERING, \
    COMFORTABLE_ACTIONS_SMALL_STEER_LEFT, COMFORTABLE_ACTIONS_SMALL_STEER_RIGHT, \
    COMFORTABLE_ACTIONS_LARGE_STEER_LEFT, COMFORTABLE_ACTIONS_LARGE_STEER_RIGHT, \
    COMFORTABLE_ACTIONS_MAINTAIN_SPEED, COMFORTABLE_ACTIONS_DECREASE_SPEED, \
    COMFORTABLE_ACTIONS_INCREASE_SPEED
from deepdrive_zero.discrete.comfortable_actions2 import \
    COMFORTABLE_ACTIONS2_IDLE, COMFORTABLE_ACTIONS2_DECAY_STEERING, \
    COMFORTABLE_ACTIONS2_SMALL_STEER_LEFT, \
    COMFORTABLE_ACTIONS2_SMALL_STEER_RIGHT, \
    COMFORTABLE_ACTIONS2_LARGE_STEER_LEFT, \
    COMFORTABLE_ACTIONS2_LARGE_STEER_RIGHT, COMFORTABLE_ACTIONS2_MAINTAIN_SPEED, \
    COMFORTABLE_ACTIONS2_DECREASE_SPEED, COMFORTABLE_ACTIONS2_INCREASE_SPEED, \
    COMFORTABLE_ACTIONS2_MICRO_STEER_LEFT, \
    COMFORTABLE_ACTIONS2_MICRO_STEER_RIGHT, COMFORTABLE_ACTIONS2
from deepdrive_zero.discrete.comfortable_steering_actions import \
    COMFORTABLE_STEERING_ACTIONS
from deepdrive_zero.experience_buffer import ExperienceBuffer
from deepdrive_zero.logs import log
from deepdrive_zero.map_gen import get_intersection
from deepdrive_zero.physics.bike_model import bike_with_friction_step, \
    get_vehicle_model, get_angle_for_accel
from deepdrive_zero.physics.collision_detection import get_rect, \
    get_lines_from_rect_points
from deepdrive_zero.physics.interpolation_state import PhysicsInterpolationState
from deepdrive_zero.physics.lane_distance import get_lane_distance
from deepdrive_zero.physics.physics_step import physics_step
from deepdrive_zero.utils import get_angles_ahead, get_angle, flatten_points, \
    np_rand, is_number


class Agent:
    def __init__(self,
                 env,
                 agent_index,
                 vehicle_width=VEHICLE_WIDTH,
                 vehicle_length=VEHICLE_LENGTH,
                 disable_gforce_penalty=False,
                 match_angle_only=False,
                 static_map=False,
                 add_rotational_friction=True,
                 add_longitudinal_friction=True,

                 # Env config params set by env
                 jerk_penalty_coeff=None,
                 gforce_penalty_coeff=None,
                 lane_penalty_coeff=None,
                 collision_penalty_coeff=None,
                 speed_reward_coeff=None,
                 win_coefficient=None,
                 gforce_threshold=None,
                 jerk_threshold=None,
                 constrain_controls=None,
                 ignore_brake=None,
                 forbid_deceleration=None,
                 expect_normalized_action_deltas=None,
                 incent_win=None,
                 dummy_accel_agent_indices=None,
                 wait_for_action=None,
                 incent_yield_to_oncoming_traffic=None,
                 physics_steps_per_observation=None,
                 end_on_lane_violation=None,
                 discrete_actions=None,
                 lane_margin=None,):

        self.env = env

        # Agent config ---------------------------------------------------------
        self.dt = env.target_dt
        self.agent_index = agent_index
        self.match_angle_only = match_angle_only
        self.static_map = static_map
        self.add_rotational_friction = add_rotational_friction
        self.add_longitudinal_friction = add_longitudinal_friction
        self.expect_normalized_actions: bool = env.expect_normalized_actions
        self.px_per_m = env.px_per_m

        # Env config params
        self.jerk_penalty_coeff = jerk_penalty_coeff
        self.gforce_penalty_coeff = gforce_penalty_coeff
        self.lane_penalty_coeff = lane_penalty_coeff
        self.collision_penalty_coeff = collision_penalty_coeff
        self.speed_reward_coeff = speed_reward_coeff
        self.win_coefficient = win_coefficient
        self.gforce_threshold = gforce_threshold
        self.jerk_threshold = jerk_threshold
        self.constrain_controls = constrain_controls
        self.ignore_brake = ignore_brake
        self.forbid_deceleration = forbid_deceleration
        self.expect_normalized_action_deltas = expect_normalized_action_deltas
        self.incent_win = incent_win
        self.dummy_accel_agent_indices = dummy_accel_agent_indices
        self.wait_for_action = wait_for_action
        self.incent_yield_to_oncoming_traffic = incent_yield_to_oncoming_traffic
        self.physics_steps_per_observation = physics_steps_per_observation
        self.end_on_lane_violation = end_on_lane_violation
        self.discrete_actions = discrete_actions
        self.lane_margin = lane_margin

        # Map type
        self.is_one_waypoint_map: bool = env.is_one_waypoint_map
        self.is_intersection_map: bool = env.is_intersection_map

        # 0.22 m/s on 0.1
        self.max_one_waypoint_mult = 0.5  # Less than 2.5 m/s on 0.1?

        # Used for old waypoint per meter map
        self.map_query_seconds_ahead: np.array = np.array(
            [0.5, 1, 1.5, 2, 2.5, 3])
        if env.is_intersection_map:
            self.num_angles_ahead = 2
        elif env.is_one_waypoint_map:
            self.num_angles_ahead = 1
        else:
            self.num_angles_ahead = len(self.map_query_seconds_ahead)

        # Map
        # These are duplicated per agent now as map is very small and most
        # map data is specific to agent
        self.map = None
        self.map_kd_tree = None
        self.map_flat = None
        self.intersection = None

        # Static obstacle
        self.add_static_obstacle: bool = env.add_static_obstacle
        self.static_obstacle_points: np.array = np.array([[0,0], [0,0]])
        self.static_obstacle_tuple: tuple = ()

        # All units in meters and radians unless otherwise specified
        self.vehicle_width: float = vehicle_width
        self.vehicle_model:List[float] = get_vehicle_model(vehicle_width)  # TODO: Use length instead. Need to retrain models to fix.
        self.vehicle_length: float = vehicle_length
        if 'STRAIGHT_TEST' in os.environ:
            self.num_actions = 1  # Accel
        else:
            self.num_actions: int = 3  # Steer, accel, brake
        self.fps: int = self.env.fps

        # Actions per second
        self.aps = self.fps / self.physics_steps_per_observation
        
        self.disable_gforce_penalty = disable_gforce_penalty
        self.observation_space = env.observation_space
        if self.constrain_controls:
            self.max_steer_change_per_tick = MAX_STEER_CHANGE_PER_SECOND / self.fps
            self.max_accel_change_per_tick = MAX_ACCEL_CHANGE_PER_SECOND / self.fps
            self.max_brake_change_per_tick = MAX_BRAKE_CHANGE_PER_SECOND / self.fps
            self.max_steer_change = self.max_steer_change_per_tick * self.physics_steps_per_observation
            self.max_accel_change = self.max_accel_change_per_tick * self.physics_steps_per_observation
            self.max_brake_change = self.max_brake_change_per_tick * self.physics_steps_per_observation
        else:
            self.max_steer_change_per_tick = STEERING_RANGE / 2
            self.max_accel_change_per_tick = MAX_METERS_PER_SEC_SQ
            self.max_brake_change_per_tick = MAX_BRAKE_G * G_ACCEL
            self.max_steer_change = self.max_steer_change_per_tick
            self.max_accel_change = self.max_accel_change_per_tick
            self.max_brake_change = self.max_brake_change_per_tick
        if discrete_actions == COMFORTABLE_STEERING_ACTIONS:
            self.convert_discrete_actions = self.convert_comfortable_steering_actions
        elif discrete_actions == COMFORTABLE_ACTIONS:
            self.convert_discrete_actions = self.convert_comfortable_actions
        elif discrete_actions == COMFORTABLE_ACTIONS2:
            self.convert_discrete_actions = self.convert_comfortable_actions2
        elif discrete_actions is not None:
            raise NotImplementedError(f'Discrete actions: {discrete_actions} not handled')

        self.update_intermediate_physics = (self.env.should_render or
                                            self.env.being_played)
        # End agent config -----------------------------------------------------

        # Agent state ----------------------------------------------------------
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

        # Prev position
        self.prev_x = None
        self.prev_y = None
        self.prev_angle = 0

        self.ego_rect: np.array = np.array(
            [0, 0] * 4)  # 4 points of ego corners
        self.ego_rect_tuple: tuple = ()  # 4 points of ego corners as tuple
        self.ego_lines: tuple = ()  # 4 edges of ego
        self.collided_with: list = []
        self.done: bool = False
        self.prev_desired_accel = 0
        self.prev_desired_steer = 0
        self.prev_desired_brake = 0
        self.other_agent_inputs: list = None
        self.static_obst_angle_info: list = None
        self.static_obst_pixels: np.array = None
        self.angle_change: float = 0
        self.prev_action: List[float] = [0] * self.num_actions
        self.prev_steer: float = 0
        self.prev_throttle: float = 0
        self.prev_brake: float = 0
        self.episode_reward: float = 0
        self.speed: float = 0
        self.prev_speed: float = 0
        self.episode_steps: int = 0
        self.num_episodes: int = 0
        self.total_steps: int = 0
        self.last_step_time: float = None
        self.wall_dt: float = None
        self.last_sleep_time: float = None
        self.total_episode_time: float = 0
        self.distance_along_route: float = None
        self.distance_traveled: float = 0
        self.distance_to_end: float = 0
        self.prev_distance_along_route: float = None
        self.furthest_distance: float = 0
        self.velocity: np.array = np.array((0, 0), dtype=np.float64)
        self.angular_velocity: float = 0
        self.gforce: float = 0  # TODO: Use accel_magnitude internally instead so we're in SI units
        self.accel_magnitude: float = 0
        self.gforce_levels: Box = self.blank_gforce_levels()
        self.max_gforce: float = 0
        self.max_jerk: float = 0
        self.state_buffer: deque = deque(maxlen=math.ceil(2*self.aps))
        self.jerk: np.array = np.array((0, 0), dtype=np.float64)  # m/s^3 instantaneous, i.e. frame to frame
        self.jerk_magnitude: float = 0
        self.closest_map_index: int = 0
        self.next_map_index: int = 1
        self.closest_waypoint_distance: float = 0
        self.waypoint_distances: np.array = np.array((0, 0), dtype=np.float64)
        self.trip_pct: float = 0
        self.avg_trip_pct: float = 0
        self._trip_pct_total: float = 0
        self.angles_ahead: List[float] = []
        self.angle_accuracies: List[float] = []
        self.episode_gforces: List[float] = []
        self.episode_jerks: List[float] = []
        self.static_obst_angle_info: list = None
        self.acceleration = np.array((0, 0), dtype=np.float64)
        self.approaching_intersection = False
        self.max_accel_historical = -np.inf
        # Important: this should only be true after the agent has reached
        # the intersection. Otherwise, we can not incent agent to actually reach
        # the intersection! We may need to create another input that
        # signals the agent is approaching a left turn (in right-hand-traffic) but
        # in theory the agent should learn to detect a left is coming
        # without an extra input explicitly stating that with RL.
        self.will_turn_across_opposing_lanes = False
        # TODO: Haven't used this in a while, needs to be updated
        # Take last n (10 from 0.5 seconds) state, action, reward values and append them
        # to the observation. Should change frame rate?
        self.experience_buffer = None
        self.should_add_previous_states = '--disable-prev-states' not in sys.argv
        self.rolling_velocity = np.array((0, 0), dtype=np.float64)
        self.rolling_accel = np.array((0, 0), dtype=np.float64)
        self.rolling_jerk = np.array((0, 0), dtype=np.float64)
        self.rolling_velocity_magnitude = 0
        self.rolling_accel_magnitude = 0
        self.rolling_jerk_magnitude = 0
        self.last_step_output = None
        self.physics_interpolation_state = PhysicsInterpolationState(
            total_steps=self.physics_steps_per_observation)
        self.step_input = None
        # End of agent state -------------------------------------------------

        self.reset()
        self.check_state()

    def check_state(self):
        # Sanity check of state setter, getter
        check_state = self.get_state()
        self.set_state(check_state)
        assert check_state == self.get_state()

    def get_state(self):
        return (self.x,
                self.y,
                self.angle,
                self.prev_x,
                self.prev_y,
                self.prev_angle,
                self.angle_to_waypoint,
                self.front_to_waypoint,
                self.start_x,
                self.start_y,
                self.start_angle,
                self.ego_rect,
                self.ego_rect_tuple,
                self.ego_lines,
                self.collided_with,
                self.done,
                self.prev_desired_accel,
                self.prev_desired_steer,
                self.prev_desired_brake,
                self.other_agent_inputs,
                self.static_obst_angle_info,
                self.static_obst_pixels,
                self.angle_change,
                self.prev_action,
                self.prev_steer,
                self.prev_throttle,
                self.prev_brake,
                self.episode_reward,
                self.speed,
                self.prev_speed,
                self.episode_steps,
                self.num_episodes,
                self.total_steps,
                self.last_step_time,
                self.wall_dt,
                self.last_sleep_time,
                self.total_episode_time,
                self.distance_along_route,
                self.distance_traveled,
                self.distance_to_end,
                self.prev_distance_along_route,
                self.furthest_distance,
                self.velocity,
                self.angular_velocity,
                self.gforce,
                self.accel_magnitude,
                self.gforce_levels,
                self.max_gforce,
                self.state_buffer,
                self.jerk,
                self.jerk_magnitude,
                self.closest_map_index,
                self.next_map_index,
                self.closest_waypoint_distance,
                self.waypoint_distances,
                self.trip_pct,
                self.avg_trip_pct,
                self._trip_pct_total,
                self.angles_ahead,
                self.angle_accuracies,
                self.episode_gforces,
                self.episode_jerks,
                self.static_obst_angle_info,
                self.acceleration,
                self.approaching_intersection,
                self.max_accel_historical,
                self.will_turn_across_opposing_lanes,
                self.experience_buffer,
                self.should_add_previous_states,
                self.rolling_velocity,
                self.rolling_accel,
                self.rolling_jerk,
                self.rolling_velocity_magnitude,
                self.rolling_accel_magnitude,
                self.rolling_jerk_magnitude,)

    def set_state(self, s):
        (self.x,
         self.y,
         self.angle,
         self.prev_x,
         self.prev_y,
         self.prev_angle,
         self.angle_to_waypoint,
         self.front_to_waypoint,
         self.start_x,
         self.start_y,
         self.start_angle,
         self.ego_rect,
         self.ego_rect_tuple,
         self.ego_lines,
         self.collided_with,
         self.done,
         self.prev_desired_accel,
         self.prev_desired_steer,
         self.prev_desired_brake,
         self.other_agent_inputs,
         self.static_obst_angle_info,
         self.static_obst_pixels,
         self.angle_change,
         self.prev_action,
         self.prev_steer,
         self.prev_throttle,
         self.prev_brake,
         self.episode_reward,
         self.speed,
         self.prev_speed,
         self.episode_steps,
         self.num_episodes,
         self.total_steps,
         self.last_step_time,
         self.wall_dt,
         self.last_sleep_time,
         self.total_episode_time,
         self.distance_along_route,
         self.distance_traveled,
         self.distance_to_end,
         self.prev_distance_along_route,
         self.furthest_distance,
         self.velocity,
         self.angular_velocity,
         self.gforce,
         self.accel_magnitude,
         self.gforce_levels,
         self.max_gforce,
         self.state_buffer,
         self.jerk,
         self.jerk_magnitude,
         self.closest_map_index,
         self.next_map_index,
         self.closest_waypoint_distance,
         self.waypoint_distances,
         self.trip_pct,
         self.avg_trip_pct,
         self._trip_pct_total,
         self.angles_ahead,
         self.angle_accuracies,
         self.episode_gforces,
         self.episode_jerks,
         self.static_obst_angle_info,
         self.acceleration,
         self.approaching_intersection,
         self.max_accel_historical,
         self.will_turn_across_opposing_lanes,
         self.experience_buffer,
         self.should_add_previous_states,
         self.rolling_velocity,
         self.rolling_accel,
         self.rolling_jerk,
         self.rolling_velocity_magnitude,
         self.rolling_accel_magnitude,
         self.rolling_jerk_magnitude,) = s

    def setup_step(self, action):
        info = Box(default_box=True)
        if 'STRAIGHT_TEST' in os.environ:
            steer = 0
            brake = 0
        elif 'FLOOR_IT' in os.environ:
            steer = 0
            brake = 0
            accel = MAX_METERS_PER_SEC_SQ
        elif 'TURN_ONE_G' in os.environ:
            _, accel, brake = action
            # steer_sign = 1 - (2 * (self.total_steps % 2))
            steer_sign = 1
            steer = steer_sign * self.get_angle_for_comfortable_turn()
            accel = 1
        else:
            if self.discrete_actions is None:
                steer, accel, brake = action
                steer, accel, brake = self.denormalize_actions(steer, accel,
                                                               brake)
            else:
                steer, accel, brake = self.convert_discrete_actions(action)

        self.prev_desired_steer = steer
        self.prev_desired_accel = accel
        self.prev_desired_brake = brake

        if '--simple-steer' in sys.argv and self.angles_ahead:
            accel = MAX_METERS_PER_SEC_SQ * 0.7
            if self.angles_ahead:
                steer = -0.1 * self.angles_ahead[0]
        elif self.match_angle_only:
            accel = MAX_METERS_PER_SEC_SQ * 0.7
            brake = 0

        if self.ignore_brake:
            brake = False
        if self.speed > 100:
            log.warning('Cutting off throttle at speed > 100m/s')
            accel = 0

        info.stats.steer = steer
        info.stats.accel = accel
        info.stats.brake = brake
        info.stats.speed = self.speed
        info.stats.episode_time = self.env.total_episode_time

        return steer, accel, brake, info

    def possibly_partial_step(self):
        action, steer, accel, brake, info = self.step_input
        now = time.time()
        if self.last_step_time is None:
            # init
            self.last_step_time = now
            reward = 0
            done = False
            observation = self.get_blank_observation()
        else:
            collided = bool(self.collided_with)
            if self.update_intermediate_physics:
                # Get intermediate physics updates between actions
                interpolation_steps = 1
                start_interpolation_index = self.physics_interpolation_state.i
                self.step_physics(steer, accel, brake, info,
                                  interpolation_steps,
                                  start_interpolation_index)
                if not self.physics_interpolation_state.ready():
                    return PARTIAL_PHYSICS_STEP
            else:
                interpolation_steps = self.physics_steps_per_observation
                start_interpolation_index = 0
                self.step_physics(steer, accel, brake, info,
                                  interpolation_steps,
                                  start_interpolation_index)

            obs_data = self.get_observation(steer, accel, brake, info)

            (closest_waypoint_distance, observation, closest_map_point,
             left_lane_distance, right_lane_distance) = obs_data

            done, won, lost = self.get_done(closest_map_point, closest_waypoint_distance,
                                            collided, info, left_lane_distance,
                                            right_lane_distance)
            reward, info = self.get_reward(
                won, lost, collided, info, steer, accel,
                left_lane_distance, right_lane_distance)
            info.stats.left_lane_distance = left_lane_distance
            info.stats.right_lane_distance = right_lane_distance
            step_time = now - self.last_step_time
            # log.trace(f'step time {round(step_time, 3)}')

            if done:
                info.stats.all_time.won = won

        return self.finish_step(action, observation, reward, done, info)

    def step(self, action):
        steer, accel, brake, info = self.setup_step(action)
        self.step_input = action, steer, accel, brake, info
        return self.possibly_partial_step()

    def finish_step(self, action, observation, reward, done, info):
        self.last_step_time = time.time()
        self.episode_reward += reward
        self.prev_action = action
        self.episode_steps += 1
        # if self.agent_index == 0:
        #     log.info(f'reward {reward}')
        # log.debug(f'accel: {round(self.accel_magnitude, 2)} jerk {round(self.jerk_magnitude, 2)}')
        # log.info(f'Speed: {round(self.speed, 2)}')
        if done:
            self.num_episodes += 1
            self._trip_pct_total += self.trip_pct
            self.avg_trip_pct = self._trip_pct_total / self.num_episodes
            episode_angle_accuracy = np.array(self.angle_accuracies).mean()
            episode_gforce_avg = np.array(self.episode_gforces).mean()
            episode_jerk_avg = np.array(self.episode_jerks).mean()
            log.debug(f'Score {round(self.episode_reward, 2)}, '
                      f'Rew/Step: {self.episode_reward / self.episode_steps}, '
                      f'Steps: {self.episode_steps}, '
                      # f'Closest map indx: {self.closest_map_index}, '
                      f'Distance {round(self.distance_along_route, 2)}, '
                      # f'Wp {self.next_map_index - 1}, '
                      f'Angular velocity {round(self.angular_velocity, 2)}, '
                      f'Speed: {round(self.speed, 2)}, '
                      f'Max gforce: {round(self.max_gforce, 4)}, '
                      f'Avg gforce: {round(episode_gforce_avg, 4)}, '
                      f'Max jerk: {round(self.max_jerk, 4)}, '
                      f'Avg jerk: {round(episode_jerk_avg, 4)}, '
                      # f'Trip pct {round(self.trip_pct, 2)}, '
                      f'Angle accuracy {round(episode_angle_accuracy, 2)}, '
                      f'Agent index {round(self.agent_index, 2)}, '
                      f'Total steps {self.total_steps}, '
                      f'Env ep# {self.env.num_episodes}, '
                      f'Ep# {self.num_episodes}')
        self.total_steps += 1
        self.set_calculated_props()
        ret = observation, reward, done, info.to_dict()
        self.last_step_output = ret
        return ret

    # TODO: Set these properties only when x,y, or angle change
    @property
    def front_x(self):
        """Front middle x position of ego"""
        theta = pi / 2 + self.angle
        return self.x + cos(theta) * self.vehicle_length / 2

    @property
    def front_y(self):
        """Front middle y position of ego"""
        theta = pi / 2 + self.angle
        return self.y + sin(theta) * self.vehicle_length / 2

    @property
    def back_x(self):
        """Back middle x position of ego"""
        theta = pi / 2 + self.angle
        return self.x - cos(theta) * self.vehicle_length / 2

    @property
    def back_y(self):
        """Back middle y position of ego"""
        theta = pi / 2 + self.angle
        return self.y - sin(theta) * self.vehicle_length / 2

    @property
    def front_pos(self):
        return np.array((self.front_x, self.front_y))

    @property
    def ego_pos(self):
        return np.array((self.x, self.y))

    @property
    def heading(self):
        return np.array([self.front_x, self.front_y]) - self.ego_pos

    # TODO: Numba this
    def denormalize_actions(self, steer, accel, brake):

        if self.expect_normalized_action_deltas:
            steer, accel, brake = self.check_action_bounds(accel, brake, steer)
            steer *= self.max_steer_change
            if self.forbid_deceleration:
                accel = self.max_accel_change * ((1 + accel) / 2)  # (-1,1) to positive only
            else:
                accel = accel * self.max_accel_change
            brake = self.max_brake_change * ((1 + brake) / 2)  # (-1,1) to positive only

            steer += self.prev_steer
            accel += self.prev_throttle
            brake += self.prev_brake

            steer = min(steer, MAX_STEER)
            steer = max(steer, MIN_STEER)

            accel = min(accel, MAX_METERS_PER_SEC_SQ)
            accel = max(accel, -MAX_METERS_PER_SEC_SQ)  # TODO: Lower max reverse accel and speed

            brake = min(brake, MAX_METERS_PER_SEC_SQ)
            brake = max(brake, 0)

        elif self.expect_normalized_actions:
            steer, accel, brake = self.check_action_bounds(accel, brake, steer)
            steer = steer * STEERING_RANGE
            if self.forbid_deceleration:
                accel = MAX_METERS_PER_SEC_SQ * ((1 + accel) / 2)  # Positive only
            else:
                accel *= MAX_METERS_PER_SEC_SQ
            brake = MAX_METERS_PER_SEC_SQ * ((1 + brake) / 2)  # Positive only
            brake = max(brake, 0)
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
        if is_blank:
            self.set_distance()

        inputs = []

        if len(angles_ahead) == 1:
            inputs += [angles_ahead[0], angles_ahead[0]]
        else:
            inputs += angles_ahead

        if self.is_intersection_map:
            # TODO: Move get_intersection_observation here

            # We avoided passing multiple waypoint distances as these get zeroed out
            # frequently, but this just leads to zero gradients on those weights and zero
            # learning rates via Adam - see play/mlp.py - so we should be able
            # to pass zero for inputs a all or a lot of the time with no problems.
            inputs.append(self.waypoint_distances[0])
            inputs.append(np.sum(self.waypoint_distances[:2]))
            inputs += self.get_other_agent_inputs(is_blank)
            inputs += list(self.velocity)
            inputs += list(self.acceleration)
            # if self.agent_index == 0:
            #     log.debug(inputsa)

        if self.env.add_static_obstacle:
            inputs += self.get_static_obstacle_inputs(is_blank)

        # if self.agent_index == 0:
        #     log.debug(
        #         f'angles ahead {angles_ahead}\n'
        #         f'prev_steer {self.prev_steer}\n'
        #         f'prev_accel {self.prev_accel}\n'
        #         f'speed {self.speed}\n'
        #         f'left {left_lane_distance}\n'
        #         f'right {right_lane_distance}\n'
        #         # f'done {done_input}\n'
        #         f'waypoint_distances {self.waypoint_distances}\n'
        #         f'velocity {self.velocity}\n'
        #         f'acceleration {self.acceleration}\n'
        #         # f'agents {other_agent_inputs}\n'
        #     )

        # TODO: These model inputs should be recorded somehow so we can
        #   use the trained models later on without needing this code.

        # TODO: Normalize these and ensure they don't exceed reasonable
        #   physical bounds
        inputs += [
            # Previous outputs (TODO: Remove for recurrent models like r2d1 / lstm / gtrxl? Deepmind R2D2 does input prev action to LSTM.)
            # self.prev_desired_steer,
            # self.prev_desired_accel,
            # self.prev_desired_brake,

            # Previous outputs after physical constraints applied
            self.prev_steer,
            self.prev_throttle,
            self.prev_brake,

            self.speed,
            # self.accel_magnitude,
            # self.jerk_magnitude,
            # self.distance_to_end,
            left_lane_distance,
            right_lane_distance,
        ]

        if self.incent_yield_to_oncoming_traffic:
            inputs.append(float(self.will_turn_across_opposing_lanes))


        # if is_blank:
        #     common_inputs = np.array(common_inputs) * 0
        # else:
        #     common_inputs = np.array(common_inputs)

        # return np.array([angles_ahead[0], self.prev_steer])

        return np.array(inputs)

        # if self.is_one_waypoint_map:
        #     if self.match_angle_only:
        #         return np.array([angles_ahead[0], self.prev_steer])
        #     elif 'STRAIGHT_TEST' in os.environ:
        #         return np.array(common_inputs)
        #     else:
        #         # One waypoint map
        #         return np.array(common_inputs)
        #         # ret = [angles_ahead[0], self.prev_steer, self.prev_accel,
        #         #        self.speed, self.distance_to_end]
        #
        # elif self.is_intersection_map:
        #     return np.array(common_inputs)
        # else:
        #     # TODO: Remove old ~100 waypoint map stuff as all driving can be
        #     #  simplified to reaching single waypoint with desired speed and
        #     #  heading (right?!?). Static and dynamic obstacles can interfere with
        #     #  ability to reach waypoint, but skipping waypoints should not
        #     #  be an immediate alternative. Training can then focus on minimizing
        #     #  g-forces if we don't reach the waypoint. Then at test time
        #     #  we can set a new waypoint after some timeout.
        #     observation = np.array(observation.values())
        #     observation = np.concatenate((observation, angles_ahead),
        #                                  axis=None)
        #
        #     if self.should_add_previous_states:
        #         if self.experience_buffer.size() == 0:
        #             self.experience_buffer.setup(shape=(len(observation),))
        #         if is_blank:
        #             past_values = np.concatenate(np.array(
        #                 self.experience_buffer.blank_buffer), axis=0)
        #         else:
        #             self.experience_buffer.maybe_add(
        #                 observation, self.env.total_episode_time)
        #             past_values = np.concatenate(np.array(
        #                 self.experience_buffer.buffer), axis=0)
        #         observation = np.concatenate((observation, past_values),
        #                                      axis=None)
        #     if math.nan in observation:
        #         raise RuntimeError(f'Found NaN in observation')
        #     if np.nan in observation:
        #         raise RuntimeError(f'Found NaN in observation')
        #     if -math.inf in observation or math.inf in observation:
        #         raise RuntimeError(f'Found inf in observation')


    def get_other_agent_inputs(self, is_blank=False):

        # TODO: Perhaps we should feed this into a transformer / LSTM / or
        #  use attention as the number of agents can be variable in length and
        #  may exceed the amount of input we want to pass to the net.
        #  Also could do max-pool like OpenAI V
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
                agent = self.env.all_agents[i]
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
        if not is_blank:
            start_static_obs = self.static_obstacle_points[0]
            end_static_obs = self.static_obstacle_points[1]
            start_obst_angle = self.get_angle_to_point(start_static_obs)
            end_obst_angle = self.get_angle_to_point(end_static_obs)
            start_obst_dist = np.linalg.norm(start_static_obs - self.front_pos)
            end_obst_dist = np.linalg.norm(end_static_obs - self.front_pos)
        else:
            start_obst_angle = 0
            end_obst_angle = 0
            start_obst_dist = 0
            end_obst_dist = 0

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
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_angle = self.angle
        self.angle_change = 0
        self.speed = 0
        self.prev_speed = 0
        self.episode_reward = 0
        self.episode_steps = 0
        self.distance_along_route = None
        self.distance_traveled = 0
        self.prev_distance_along_route = None
        self.furthest_distance = 0
        self.velocity = np.array((0, 0), dtype=np.float64)
        self.angular_velocity = 0
        self.acceleration = np.array((0, 0), dtype=np.float64)
        self.gforce = 0
        self.accel_magnitude = 0
        self.jerk = 0
        self.jerk_magnitude = 0
        self.gforce_levels = self.blank_gforce_levels()
        self.max_gforce = 0
        self.max_jerk = 0
        self.closest_map_index = 0
        self.next_map_index = 1
        self.trip_pct = 0
        self.angles_ahead = []
        self.angle_accuracies = []
        self.episode_gforces = []
        self.episode_jerks = []
        self.collided_with = []
        self.done = False
        self.prev_throttle = 0
        self.prev_steer = 0
        self.prev_brake = 0

        # TODO: Regen map every so often
        if self.map is None or not self.static_map:
            self.gen_map()

        self.set_calculated_props()

        if self.experience_buffer is None:
            self.experience_buffer = ExperienceBuffer()
        self.experience_buffer.reset()
        self.state_buffer.clear()
        self.rolling_velocity = np.array((0, 0), dtype=np.float64)
        self.rolling_accel = np.array((0, 0), dtype=np.float64)
        self.rolling_jerk = np.array((0, 0), dtype=np.float64)
        self.rolling_velocity_magnitude = 0
        self.rolling_accel_magnitude = 0
        self.rolling_jerk_magnitude = 0
        obz = self.get_blank_observation()

        return obz

    def set_calculated_props(self):
        self.ego_rect, self.ego_rect_tuple = get_rect(
            self.x, self.y, self.angle, self.vehicle_width, self.vehicle_length)

        self.ego_lines = get_lines_from_rect_points(self.ego_rect_tuple)

    def get_done(self, closest_map_point, lane_deviation,
                 collided: bool, info: Box,
                 left_lane_distance: float,
                 right_lane_distance: float) -> Tuple[bool, bool, bool]:
        done = False
        won = False
        lost = False
        info.stats.done_only.collided = 0
        info.done_only.harmful_gs = 0
        info.done_only.harmful_jerk = 0
        info.stats.done_only.timeup = 0
        info.stats.done_only.exited_lane = 0
        info.stats.done_only.circles = 0
        info.stats.done_only.skipped = 0
        info.stats.done_only.backwards = 0
        info.stats.done_only.won = 0

        if 'DISABLE_GAME_OVER' in os.environ:
            return done, won, lost
        elif collided:
            log.warning(f'Collision, game over agent {self.agent_index}')
            info.stats.done_only.collided = 1
            done = True
            lost = True
        elif self.gforce_threshold and self.gforce > self.gforce_threshold:
            # Only end on g-force once we've learned to complete part of the trip.
            log.warning(f'Harmful g-forces, game over agent {self.agent_index}')
            info.done_only.harmful_gs = 1
            done = True
            lost = True
        elif self.jerk_threshold and self.jerk_magnitude > self.jerk_threshold:
            # Only end on g-force once we've learned to complete part of the trip.
            log.warning(f'Harmful jerk, game over agent {self.agent_index}')
            info.done_only.harmful_jerk = 1
            done = True
            lost = True
        elif self.end_on_lane_violation and (right_lane_distance < -0.25
                                             or left_lane_distance < -0.25):
            log.warning(f'Exited lane, game over agent {self.agent_index}')
            info.done_only.exited_lane = 1
            done = True
            lost = True
        elif (self.episode_steps + 1) % self.env._max_episode_steps == 0:
            info.stats.done_only.timeup = 1
            log.warning(f"Time's up agent {self.agent_index}")
            done = True
        elif 'DISABLE_CIRCLE_CHECK' not in os.environ and \
                abs(math.degrees(self.angle)) > 400:
            info.stats.done_only.circles = 1
            done = True
            lost = True
            log.warning(f'Going in circles - angle {math.degrees(self.angle)} too high')
        elif self.is_one_waypoint_map or self.is_intersection_map:
            if self.is_intersection_map and \
                    self.closest_map_index > self.next_map_index:
                # Negative progress catches this first depending on distance
                # thresholds
                info.stats.done_only.skipped = 1
                done = True
                lost = True
                log.warning(f'Skipped waypoint {self.next_map_index} '
                            f'agent {self.agent_index}')
            elif (self.furthest_distance - self.distance_along_route) > 2:
                info.stats.done_only.backwards = 1
                done = True
                lost = True
                log.warning(f'Negative progress agent {self.agent_index}')
            elif abs(self.map.route_length - self.distance_along_route) < 1:
                info.stats.done_only.won = 1
                done = True
                won = True
                # You win!
                log.success(f'Reached destination! '
                            f'Steps: {self.episode_steps} '
                            f'Agent: {self.agent_index}')
        elif list(self.map.waypoints[-1]) == list(closest_map_point):
            # You win!
            info.stats.done_only.won = 1
            done = True
            won = True
            log.success(f'Reached destination! '
                        f'Steps: {self.episode_steps}')
        if '--test-win' in sys.argv:
            won = True
        self.done = done
        return done, won, lost


    def get_reward(self, won: bool, lost: bool,
                   collided: bool, info: Box, steer: float,
                   accel: float, left_lane_distance: float,
                   right_lane_distance: float) -> Tuple[float, Box]:

        angle_diff = abs(self.angles_ahead[0])

        if 'STRAIGHT_TEST' in os.environ:
            angle_reward = 0
        else:
            angle_reward = 4 * pi - angle_diff

        angle_accuracy = 1 - angle_diff / (2 * pi)
        if self.agent_index == 0:
            # log.debug(f'angle accuracy {angle_accuracy}')
            pass

        speed_reward = self.get_progress_reward()
        if (self.incent_yield_to_oncoming_traffic and
                self.will_turn_across_opposing_lanes and
                self.upcoming_opposing_lane_agents()):
            # Even if other agent is turning across as well, and so won't
            # intersect (i.e. both agents turning left in right hand traffic)
            # we should be cautious while making the left.
            # Once the other agent has passed, we get distance
            # reward for the turn which should incent being less cautious
            # when no other agent is approaching.
            speed_reward = -abs(speed_reward)
            # log.info('TURNING ACROSS')

        # TODO: Idea penalize residuals of a quadratic regression fit to history
        #  of actions. Currently penalizing jerk instead which may or may not
        #  be better (but it is simpler).
        if 'ACTION_PENALTY' in os.environ:
            action_penalty = float(os.environ['ACTION_PENALTY'])
            steer_penalty = abs(self.prev_steer - steer) * action_penalty
            accel_penalty = abs(self.prev_throttle - accel) * action_penalty
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

        # lane_penalty = 0
        # if left_lane_distance < 0.7:
        #     lane_penalty += (left_lane_distance + 1)**2
        # if right_lane_distance < 0.7:
        #     # yes both can happen if you're orthogonal to the lane
        #     lane_penalty += (right_lane_distance + 1)**2
        # lane_penalty *= self.lane_penalty_coeff

        # """
        # lane_penalty = 0
        # if left_lane_distance < 0.7:
        #     lane_penalty += (left_lane_distance + 1)**2
        # if right_lane_distance < 0.7:
        #     # yes both can happen if you're orthogonal to the lane
        #     lane_penalty += (right_lane_distance + 1)**2
        # lane_penalty *= self.lane_penalty_coeff
        # """
        lane_penalty = 0
        lane_margin = self.lane_margin  # Consider increasing outside intersection
        left_lane_margin_dist = left_lane_distance - lane_margin
        if left_lane_margin_dist < 0:
            lane_penalty += abs(left_lane_margin_dist)
        right_lane_margin_dist = right_lane_distance - lane_margin
        if right_lane_margin_dist < 0:
            # yes both can happen if you're orthogonal to the lane
            lane_penalty += abs(right_lane_margin_dist)
        lane_penalty *= self.lane_penalty_coeff

        # if self.agent_index == 0:
        #     log.debug(f'lane penalty {lane_penalty} {lane_penalty2}')

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
        #         # f'next waypoint {self.next_map_index} '
        #         f'distance {self.distance_along_route} '
        #         # f'rew/dist {round(ret/(self.distance - self.prev_distance),3)} '
        #         f'speed {speed_reward} '
        #         f'gforce {accel_penalty} '
        #         f'jerk {jerk_penalty} '
        #         f'lane {lane_penalty} '
        #         f'win {win_reward} '
        #         f'angle_accuracy {angle_accuracy} '
        #         # f'collision {collision_penalty} '
        #         # f'steer {steer_penalty} '
        #         # f'accel {accel_penalty} '
        #     )

        return ret, info

    def get_progress_reward(self):
        frame_distance = self.distance_along_route - self.prev_distance_along_route
        speed_reward = frame_distance * self.speed_reward_coeff
        return speed_reward

    def get_win_reward(self, won):
        win_reward = 0
        if self.incent_win and won:
            win_reward = self.win_coefficient
        return win_reward

    def get_observation(self, steer, accel, brake, info):

        closest_map_point, closest_map_index, closest_waypoint_distance = \
            get_closest_point((self.front_x, self.front_y), self.map_kd_tree)

        self.closest_waypoint_distance = closest_waypoint_distance
        self.closest_map_index = closest_map_index
        self.set_distance()

        self.trip_pct = 100 * self.distance_along_route / self.map.route_length

        half_lane_width = self.map.lane_width / 2
        left_lane_distance = right_lane_distance = half_lane_width

        if self.is_one_waypoint_map:
            angles_ahead = self.get_one_waypoint_angle_ahead()

            # log.info(f'angle ahead {math.degrees(angles_ahead[0])}')
            # log.info(f'angle {math.degrees(self.angle)}')
        elif self.is_intersection_map:
            # angles_ahead, left_lane_distance1, right_lane_distance1, = \
            #     self.get_intersection_observation(half_lane_width,
            #                                       left_lane_distance,
            #                                       right_lane_distance)
            angles_ahead, left_lane_distance2, right_lane_distance2, = \
                self.get_intersection_observation(half_lane_width,
                                                  left_lane_distance,
                                                  right_lane_distance)
            # if self.agent_index == 0:
            #     log.trace(f'a {round(math.degrees(angles_ahead[0]), 2)} {round(math.degrees(angles_ahead2[0]), 2)}')
            #     log.trace(f'l {round(left_lane_distance1, 2)} {round(left_lane_distance2, 2)}')
            #     log.trace(f'r {round(right_lane_distance1, 2)} {round(right_lane_distance2, 2)}')

            left_lane_distance = left_lane_distance2
            right_lane_distance = right_lane_distance2

        else:
            self.trip_pct = 100 * closest_map_index / (len(self.map.waypoints) - 1)
            angles_ahead = self.get_angles_ahead(closest_map_index)

        if self.agent_index == 0:
            # log.trace(f'lane dist left {left_lane_distance} right {right_lane_distance}')
            pass

        self.angles_ahead = angles_ahead

        info.stats.closest_map_index = closest_map_index
        info.stats.done_only.trip_pct = self.trip_pct
        info.stats.distance = self.distance_along_route

        observation = self.populate_observation(
            closest_map_point=closest_map_point,
            lane_deviation=0,
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
        self.will_turn_across_opposing_lanes = False
        self.approaching_intersection = False

        (back_left, back_right, front_left, front_right, max_ego_x, max_ego_y,
         min_ego_x, min_ego_y) = self.get_rect_coords_info()

        if self.agent_index == 0:
            # Left turn agent
            intersection_start_y = mp.waypoints[1][1]
            intersection_end_x = mp.waypoints[2][0]
            if self.front_y < intersection_start_y:
                # Before entering intersection
                wp_x = mp.waypoints[0][0]
                left_lane_x = wp_x - half_lane_width
                right_lane_x = wp_x + half_lane_width
                left_distance = min_ego_x - left_lane_x
                right_distance = right_lane_x - max_ego_x
            elif self.front_x < intersection_end_x:
                # Exiting intersection
                wp_y = mp.waypoints[2][1]
                bottom_lane_y = wp_y - half_lane_width
                top_lane_y = wp_y + half_lane_width
                if self.back_x < intersection_end_x:  # intersection end, x coord
                    # Completely exited intersection
                    left_distance = min_ego_y - bottom_lane_y
                    right_distance = top_lane_y - max_ego_y
                else:
                    # Partially exited, front has exited but back has not
                    if RIGHT_HAND_TRAFFIC:
                        front_y = front_left[1], front_right[1]
                        left_distance = min(front_y) - bottom_lane_y
                        right_distance = top_lane_y - max(front_y)
            else:
                # Inside the intersection
                self.will_turn_across_opposing_lanes = True
                # Default lane reward as there are no lanes in intersection.
                # TODO: Discourage cutting into lanes when partially in the
                #   intersection.
        else:
            # Straight agent
            wp_x = mp.waypoints[0][0]
            left_lane_x = wp_x - half_lane_width
            right_lane_x = wp_x + half_lane_width
            right_distance = min_ego_x - left_lane_x
            left_distance = right_lane_x - max_ego_x
            lane_lines, _lane_width = self.intersection
            (left_vert, mid_vert, right_vert, top_horiz, mid_horiz,
             bottom_horiz) = lane_lines
            if min_ego_y > top_horiz[0][1]:  # any y coordinate will do
                self.approaching_intersection = True

        # if self.agent_index == 0:
        #     log.trace(f'across: {self.will_turn_across_opposing_lanes}\t'
        #               f'l {round(left_distance, 2)}\t'
        #               f'r {round(right_distance, 2)}')
        # else:
        #     log.trace(f'approach: {self.approaching_intersection}\t'
        #              f'l {round(left_distance, 2)}\t'
        #              f'r {round(right_distance, 2)}')

        return angles_ahead, left_distance, right_distance


    def get_intersection_observation2(self, half_lane_width, left_distance,
                                      right_distance):
        # TODO: Move other agent observations (like distances) here
        a2w = self.get_angle_to_point
        wi = self.next_map_index
        mp = self.map
        angles_ahead = [a2w(p) for p in mp.waypoints[wi:wi+2]]
        self.will_turn_across_opposing_lanes = False
        self.approaching_intersection = False

        (back_left, back_right, front_left, front_right, max_ego_x, max_ego_y,
         min_ego_x, min_ego_y) = self.get_rect_coords_info()

        # TODO: For lane distance on arbitrary maps:
        #  For each ego rect point, use a kd-tree to get the two closest
        #  lane line points.
        #    Then pass the line segment made up by those two points
        #    to get_lane_distance() with just the one
        #    ego point.
        #  Finally choose the min left and right distance from all points.
        #  The map should therefore have a string of points for each side of the
        #  lane that are equidistant at 1m apart.
        if self.agent_index == 0:
            # Left turn agent
            # N.B. Lane within intersection is diagonal
            # https://user-images.githubusercontent.com/181225/81218014-5b6ba580-8f92-11ea-8c70-e65e27c9d5f4.jpeg
            intersection_start_y = mp.waypoints[1][1]
            intersection_end_x = mp.waypoints[2][0]
            intersection_middle_x = mp.waypoints[1][0] - half_lane_width
            intersection_middle_lane_y = mp.waypoints[2][1] - half_lane_width
            min_right_distance = math.inf
            min_left_distance = math.inf
            reached_upper_left_half = False
            inside_lower_right_half = False
            for pt in self.ego_rect:
                # Assumes that we go from bottom and take left. Need proper
                # 1m spaced map points per above to generalize lane distance
                if pt[1] <= intersection_start_y:
                    # Before intersection
                    wp_x = mp.waypoints[0][0]
                    left_lane_x = wp_x - half_lane_width
                    min_left_distance = min(min_left_distance, pt[0] - left_lane_x)
                    wp_x = mp.waypoints[0][0]
                    right_lane_x = wp_x + half_lane_width
                    min_right_distance = min(min_right_distance, right_lane_x - pt[0])
                elif pt[1] <= intersection_middle_lane_y:
                    # Lower half of intersection
                    diag_left_top = \
                        (mp.waypoints[2][0], intersection_middle_lane_y)
                    diag_left_bottom = \
                        (mp.waypoints[1][0] - half_lane_width, mp.waypoints[1][1])
                    min_left_distance = min(min_left_distance, get_lane_distance(
                        p0=diag_left_bottom,
                        p1=diag_left_top,
                        ego_rect_pts=pt.reshape(1,2),
                        is_left_lane_line=True,))
                    wp_x = mp.waypoints[0][0]
                    right_lane_x = wp_x + half_lane_width
                    min_right_distance = min(min_right_distance, right_lane_x - pt[0])
                    inside_lower_right_half = True
                elif pt[0] >= intersection_middle_x:
                    # Upper right half of intersection
                    diag_right_top = \
                        (intersection_middle_x, mp.waypoints[2][1] + half_lane_width)
                    diag_right_bottom = \
                        (mp.waypoints[1][0] + half_lane_width, intersection_middle_lane_y)
                    min_right_distance = min(min_right_distance, get_lane_distance(
                        p0=diag_right_bottom,
                        p1=diag_right_top,
                        ego_rect_pts=pt.reshape(1,2),
                        is_left_lane_line=False,))
                    wp_y = mp.waypoints[2][1]
                    bottom_lane_y = wp_y - half_lane_width
                    min_left_distance = min(min_left_distance, pt[1] - bottom_lane_y)
                    inside_lower_right_half = True
                elif pt[0] >= intersection_end_x:
                    # Upper left half of intersection
                    wp_y = mp.waypoints[2][1]
                    bottom_lane_y = wp_y - half_lane_width
                    min_left_distance = min(min_left_distance, pt[1] - bottom_lane_y)
                    top_lane_y = wp_y + half_lane_width
                    min_right_distance = min(min_right_distance, top_lane_y - pt[1])
                    reached_upper_left_half = True
                elif pt[0] < intersection_end_x:
                    # Exited intersection
                    wp_y = mp.waypoints[2][1]
                    bottom_lane_y = wp_y - half_lane_width
                    top_lane_y = wp_y + half_lane_width
                    min_left_distance = min(min_left_distance, pt[1] - bottom_lane_y)
                    min_right_distance = min(min_right_distance, top_lane_y - pt[1])
                if min_left_distance != math.inf:
                    left_distance = float(min_left_distance)  # float64 causing issues?
                if min_right_distance != math.inf:
                    right_distance = float(min_right_distance)   # float64 causing issues?

                self.will_turn_across_opposing_lanes = \
                    (inside_lower_right_half and not reached_upper_left_half)
            # log.debug(f'left dist {left_distance} right dist {right_distance}')
        else:
            # Straight agent
            wp_x = mp.waypoints[0][0]
            left_lane_x = wp_x - half_lane_width
            right_lane_x = wp_x + half_lane_width
            right_distance = min_ego_x - left_lane_x
            left_distance = right_lane_x - max_ego_x
            lane_lines, _lane_width = self.intersection
            (left_vert, mid_vert, right_vert, top_horiz, mid_horiz,
             bottom_horiz) = lane_lines
            if min_ego_y > top_horiz[0][1]:  # any y coordinate will do
                self.approaching_intersection = True

        # if self.agent_index == 0:
        #     log.trace(f'across: {self.will_turn_across_opposing_lanes}\t'
        #               f'l {round(left_distance, 2)}\t'
        #               f'r {round(right_distance, 2)}')
        # else:
        #     log.trace(f'approach: {self.approaching_intersection}\t'
        #              f'l {round(left_distance, 2)}\t'
        #              f'r {round(right_distance, 2)}')

        return angles_ahead, left_distance, right_distance

    def get_rect_coords_info(self):
        x_coords = self.ego_rect.T[0]
        min_ego_x = min(x_coords)
        max_ego_x = max(x_coords)

        y_coords = self.ego_rect.T[1]
        min_ego_y = min(y_coords)
        max_ego_y = max(y_coords)

        front_coords = self.ego_rect[:2]
        back_coords = self.ego_rect[2:]

        # Clockwise
        front_left = front_coords[0]
        front_right = front_coords[1]
        back_right = back_coords[0]
        back_left = back_coords[1]

        return (back_left, back_right, front_left, front_right, max_ego_x,
                max_ego_y, min_ego_x, min_ego_y)

    def get_angles_ahead(self, closest_map_index):
        """
        Note: this assumes we are on the old meter per waypoint map path.
        This meter per waypoint map is not being used for current training
        but we may want to bring it back as we are able to achieve more complex
        behavior.
        """
        distances = self.map.distances

        return get_angles_ahead(total_points=len(distances),
                                route_length=self.map.route_length,
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

        self.prev_distance_along_route = self.distance_along_route
        if 'STRAIGHT_TEST' in os.environ:
            self.distance_along_route = self.x - self.start_x
        elif self.is_one_waypoint_map:
            end = np.array([mp.x[-1], mp.y[-1]])

            # TODO: Use self.front_pos
            pos = np.array([self.front_x, self.front_y])
            self.distance_to_end = np.linalg.norm(end - pos)

            self.distance_along_route = mp.route_length - self.distance_to_end
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
            self.distance_along_route = (mp.distances[self.next_map_index] -
                                         abs(waypoint_distances[0]))
            self.waypoint_distances = waypoint_distances
            # log.debug(waypoint_distances)
        else:
            # Assumes waypoints are very close, i.e. 1m apart
            self.distance_along_route = mp.distances[self.closest_map_index]
        # log.debug(f'distance {self.distance}')
        self.furthest_distance = max(self.distance_along_route, self.furthest_distance)
        if self.prev_distance_along_route is None:
            # Init prev distance
            self.prev_distance_along_route = self.distance_along_route

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
                       route_length=distances[-1],
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
        self.intersection = (lines, lane_width)
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
            # wps.append((mid_vert[0][0] - lane_width / 2, 30.139197872452702))
            # wps.append((mid_vert[0][0] - lane_width / 2, 15.139197872452702))
            wps.append((mid_vert[0][0] - lane_width / 2, 4.139197872452702))
        else:
            raise NotImplementedError('More than 2 agents not yet supported')

        x, y = np.array(list(zip(*wps)))

        return x, y, lane_width, lines

    def step_physics(self, steer, accel, brake, info, interpolation_steps,
                     start_interpolation_index):
        dt = self.dt
        start = time.time()
        self.prev_speed = self.speed
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_angle = self.angle

        self.update_physics(steer, accel, brake, interpolation_steps,
                            start_interpolation_index)

        # self.compute_rolling_state()

        #
        # if self.agent_index == 0:
        #     log.debug(f'accel: {self.accel_magnitude} jerk: {self.jerk_magnitude} distance_traveled {self.distance_traveled}')

        info.stats.gforce = self.gforce
        self.total_episode_time += dt * interpolation_steps
        self.env.total_episode_time += dt * interpolation_steps


        self.ego_rect, self.ego_rect_tuple = get_rect(
            self.x, self.y, self.angle, self.vehicle_width, self.vehicle_length)

        if self.physics_interpolation_state.ready():
            self.episode_gforces.append(self.gforce)
            self.episode_jerks.append(self.jerk_magnitude)

    def update_physics(self, steer, throttle, brake, interpolation_steps,
                       start_interpolation_index=0):
        (self.acceleration,
         self.angle,
         self.angle_change,
         self.angular_velocity,
         self.gforce,
         self.jerk,
         max_gforce,
         self.max_jerk,
         self.speed,
         self.x,
         self.y,
         self.prev_throttle,
         self.prev_brake,
         self.prev_steer,
         self.velocity,
         self.distance_traveled) = physics_step(
            throttle=throttle,
            add_longitudinal_friction=self.add_longitudinal_friction,
            add_rotational_friction=self.add_rotational_friction,
            brake=brake,
            curr_acceleration=self.acceleration,
            jerk=self.jerk,
            curr_angle=self.angle,
            curr_angle_change=self.angle_change,
            curr_angular_velocity=self.angular_velocity,
            curr_gforce=self.gforce,
            curr_max_gforce=self.max_gforce,
            curr_max_jerk=self.max_jerk,
            curr_speed=self.speed,
            curr_velocity=self.velocity,
            curr_x=self.x,
            curr_y=self.y,
            dt=self.dt,
            interpolation_steps=interpolation_steps,
            prev_throttle=self.prev_throttle,
            prev_brake=self.prev_brake,
            prev_steer=self.prev_steer,
            steer=steer,
            vehicle_model=self.vehicle_model,
            ignore_brake=self.ignore_brake,
            constrain_controls=self.constrain_controls,
            max_steer_change=self.max_steer_change,
            max_throttle_change=self.max_accel_change,
            max_brake_change=self.max_brake_change,
            distance_traveled=self.distance_traveled,
            start_interpolation_index=start_interpolation_index,
            interpolation_range=self.physics_steps_per_observation,)
        if self.update_intermediate_physics:
            self.physics_interpolation_state.update()

        if max_gforce > self.max_gforce:
            # log.warning(f'New max g {max_gforce}')
            self.max_gforce = max_gforce


    def compute_rolling_state(self):
        self.state_buffer.append(
            ([self.x, self.y], self.velocity, self.acceleration))
        state_len = self.state_buffer.maxlen
        if len(self.state_buffer) == state_len:
            state_array = np.array(list(self.state_buffer))
            prev_second_avg = state_array[:state_len // 2].T.mean(axis=2)
            curr_second_avg = state_array[state_len // 2:].T.mean(axis=2)
            curr_state = np.array(self.state_buffer[-1]).T
            change_between_seconds = curr_state - prev_second_avg
            rolling_velocity = change_between_seconds.T[0]
            rolling_accel = change_between_seconds.T[1]
            rolling_jerk = change_between_seconds.T[2]
            self.rolling_velocity = rolling_velocity
            self.rolling_accel = rolling_accel
            self.rolling_jerk = rolling_jerk
            self.rolling_velocity_magnitude = np.linalg.norm(
                self.rolling_velocity)
            self.rolling_accel_magnitude = np.linalg.norm(self.rolling_accel)
            self.rolling_jerk_magnitude = np.linalg.norm(self.rolling_jerk)
            # log.debug(f'v_avg: {round(average_velocity_mag, 2)} v_ins: {round(np.linalg.norm(self.velocity), 2)}')
            # log.debug(f'a_avg: {round(average_accel_mag, 2)} a_ins: {round(self.accel_magnitude, 2)}')
            # log.trace(
            #     # f'v: {round(self.rolling_velocity_magnitude, 2)} '
            #     f'a_roll: {round(self.rolling_accel_magnitude, 2)} a_ins: {round(self.accel_magnitude, 2)} '
            #     f'j: {round(self.rolling_jerk_magnitude, 2)} ')

    def upcoming_opposing_lane_agents(self) -> bool:
        if (self.agent_index == 0 and
                len(self.env.agents) > 1 and
                self.env.agents[1].approaching_intersection):
            # TODO: Make this more general when adding more agents,
            #  larger maps, etc... i.e. approaching intersection will have to
            #  identify some intersection id instead of just being a bool.
            return True
        else:
            return False

    def convert_comfortable_actions(self, action):
        comfort_accel = 1  # Comfortable g-force is around 0.1 = 1 m/s**2
        one_degree = 0.0174533
        steer, throttle, brake = 0,0,0
        action = int(action)

        debug_agent_index = 1

        # TODO: Adjust all of these to depend on actions per second
        #   NN learns to adjust, but we should decay steering for example in
        #   ~1.5 seconds, which with 0.9 ** x at 5 aps, is around 2 seconds.
        if action == COMFORTABLE_ACTIONS_IDLE:
            log.debug('idle') if self.agent_index == debug_agent_index else None
        # Steer
        elif action in COMFORTABLE_ACTIONS_DECAY_STEERING:
            # log.debug('decay steer') if self.agent_index == debug_agent_index else None
            steer = 0.9 * self.prev_steer
        elif action in COMFORTABLE_ACTIONS_SMALL_STEER_LEFT:
            # log.debug('1 deg left') if self.agent_index == debug_agent_index else None
            steer = one_degree
        elif action in COMFORTABLE_ACTIONS_SMALL_STEER_RIGHT:
            # log.debug('1 deg right') if self.agent_index == debug_agent_index else None
            steer = -one_degree
        elif action in COMFORTABLE_ACTIONS_LARGE_STEER_LEFT:
            # log.debug('large left') if self.agent_index == debug_agent_index else None
            steer = self.get_angle_for_comfortable_turn()
        elif action in COMFORTABLE_ACTIONS_LARGE_STEER_RIGHT:
            # log.debug('large right') if self.agent_index == debug_agent_index else None
            steer = -self.get_angle_for_comfortable_turn()

        # Accel
        if action in COMFORTABLE_ACTIONS_MAINTAIN_SPEED:
            # log.debug('maintain') if self.agent_index == debug_agent_index else None
            if self.prev_speed < self.speed:
                throttle = self.prev_throttle * 0.99
            elif self.prev_speed > self.speed:
                throttle = self.prev_throttle * 1.01
            else:
                throttle = self.prev_throttle
        elif action in COMFORTABLE_ACTIONS_DECREASE_SPEED:
            # log.debug('slower') if self.agent_index == debug_agent_index else None
            throttle = 0
            if self.prev_throttle == 0:
                brake = comfort_accel
        elif action in COMFORTABLE_ACTIONS_INCREASE_SPEED:
            # log.debug('faster') if self.agent_index == debug_agent_index else None
            throttle = comfort_accel
            # throttle = min(throttle, MAX_METERS_PER_SEC_SQ, comfort_accel * self.dt)

        return steer, throttle, brake

    def convert_comfortable_actions2(self, action):
        # noinspection DuplicatedCode
        comfort_accel = 1  # Comfortable g-force is around 0.1 = 1 m/s**2
        one_degree = 0.0174533
        one_tenth_degree = one_degree / 10
        steer, throttle, brake = 0,0,0
        action = int(action)
        # action = 18
        debug_agent_index = 1

        # TODO: Adjust all of these to depend on actions per second
        #   NN learns to adjust, but we should decay steering for example in
        #   ~1.5 seconds, which with 0.9 ** x at 5 aps, is around 2 seconds.
        if action == COMFORTABLE_ACTIONS2_IDLE:
            # log.debug('idle') if self.agent_index == debug_agent_index else None
            pass
        # Steer
        elif action in COMFORTABLE_ACTIONS2_DECAY_STEERING:
            # log.debug('decay steer') if self.agent_index == debug_agent_index else None
            steer = 0.9 * self.prev_steer
        elif action in COMFORTABLE_ACTIONS2_MICRO_STEER_LEFT:
            # log.debug('0.1 deg left') if self.agent_index == debug_agent_index else None
            steer = one_tenth_degree
        elif action in COMFORTABLE_ACTIONS2_MICRO_STEER_RIGHT:
            # log.debug('0.1 deg right') if self.agent_index == debug_agent_index else None
            steer = -one_tenth_degree
        elif action in COMFORTABLE_ACTIONS2_SMALL_STEER_LEFT:
            # log.debug('1 deg left') if self.agent_index == debug_agent_index else None
            steer = one_degree
        elif action in COMFORTABLE_ACTIONS2_SMALL_STEER_RIGHT:
            # log.debug('1 deg right') if self.agent_index == debug_agent_index else None
            steer = -one_degree
        elif action in COMFORTABLE_ACTIONS2_LARGE_STEER_LEFT:
            # log.debug('large left') if self.agent_index == debug_agent_index else None
            steer = self.get_angle_for_comfortable_turn()
        elif action in COMFORTABLE_ACTIONS2_LARGE_STEER_RIGHT:
            # log.debug('large right') if self.agent_index == debug_agent_index else None
            steer = -self.get_angle_for_comfortable_turn()

        # Accel
        if action in COMFORTABLE_ACTIONS2_MAINTAIN_SPEED:
            # log.debug('maintain') if self.agent_index == debug_agent_index else None
            if self.prev_speed < self.speed:
                throttle = self.prev_throttle * 0.99
            elif self.prev_speed > self.speed:
                throttle = self.prev_throttle * 1.01
            else:
                throttle = self.prev_throttle
        elif action in COMFORTABLE_ACTIONS2_DECREASE_SPEED:
            # log.debug('slower') if self.agent_index == debug_agent_index else None
            throttle = 0
            if self.prev_throttle == 0:
                brake = comfort_accel
        elif action in COMFORTABLE_ACTIONS2_INCREASE_SPEED:
            # log.debug('faster') if self.agent_index == debug_agent_index else None
            throttle = comfort_accel
            # throttle = min(throttle, MAX_METERS_PER_SEC_SQ, comfort_accel * self.dt)

        return steer, throttle, brake

    def convert_comfortable_steering_actions(self, action):
        comfort_accel = 1  # Comfortable g-force is around 0.1 = 1 m/s**2
        one_degree = 0.0174533
        steer, throttle, brake = 0,0,0
        if action == 0:
            # Idle
            # log.debug('idle')
            steer = 0
        elif action == 1:
            # Decay steering
            # log.debug('decay!')
            steer = 0.9 * self.prev_steer
        elif action == 2:
            # Small steer left
            # log.debug('small steer left!')
            steer = one_degree
        elif action == 3:
            # Small steer right
            # log.debug('small steer right!')
            steer = -one_degree
        elif action == 4:
            # Large steer left
            # log.debug('large steer left!')
            steer = self.get_angle_for_comfortable_turn(comfort_accel)
        elif action == 5:
            # Large steer right
            # log.debug('large steer right!')
            steer = -self.get_angle_for_comfortable_turn(comfort_accel)

        throttle = 0.1
        return steer, throttle, brake

    def get_angle_for_comfortable_turn(self):
        # return get_angle_for_accel(
        #     speed=self.speed,
        #     vehicle_model=self.vehicle_model,
        #     prev_angle_change=self.angle_change, aps=self.aps, angle=self.angle,
        #     prev_accel=self.accel_magnitude)
        # TODO: Automate this tuning with linear regression or analytically solve
        #   with inverse kinematics for acceleration
        return pi / (30 * max(self.speed, 1) ** 2)


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