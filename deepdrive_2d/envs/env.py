import math
import json
import os
import sys
import time
from collections import deque
from math import pi, cos, sin
from typing import Tuple, List
import random
import arcade
import gym
import numpy as np
from box import Box
from gym import spaces

from numba import njit
from retry import retry
from scipy import spatial
import pyglet

from deepdrive_2d.collision_detection import lines_intersect, check_collision, \
    get_rect
from deepdrive_2d.constants import USE_VOYAGE, MAP_WIDTH_PX, MAP_HEIGHT_PX, \
    SCREEN_MARGIN, VEHICLE_HEIGHT, VEHICLE_WIDTH, PX_PER_M, \
    MAX_METERS_PER_SEC_SQ
from deepdrive_2d.experience_buffer import ExperienceBuffer
from deepdrive_2d.map_gen import gen_map
from deepdrive_2d.utils import flatten_points, get_angles_ahead, \
    angle_between_points, angle_between_vectors
from deepdrive_2d.logs import log

MAX_BRAKE_G = 1
G_ACCEL = 9.80665
CONTINUOUS_REWARD = True
GAME_OVER_PENALTY = -1
IS_DEBUG_MODE = getattr(sys, 'gettrace', None)


class Deepdrive2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 vehicle_width=VEHICLE_WIDTH,
                 vehicle_height=VEHICLE_HEIGHT,
                 px_per_m=PX_PER_M,
                 add_rotational_friction=True,
                 add_longitudinal_friction=True,
                 return_observation_as_array=True,
                 seed_value=0,
                 ignore_brake=True,
                 expect_normalized_actions=True,
                 decouple_step_time=True,
                 physics_steps_per_observation=6,
                 one_waypoint_map=False,
                 match_angle_only=False,
                 incent_win=False,
                 gamma=0.99,
                 add_static_obstacle=False):

        # All units in meters and radians unless otherwise specified
        self.vehicle_width: float = vehicle_width
        self.vehicle_height: float = vehicle_height
        self.return_observation_as_array: bool = return_observation_as_array
        self.px_per_m: float = px_per_m
        self.ignore_brake: bool = ignore_brake
        self.expect_normalized_actions: bool = expect_normalized_actions
        self.seed_value: int = seed_value
        self.add_rotational_friction: bool = add_rotational_friction
        self.add_longitudinal_friction: bool = add_longitudinal_friction
        self.static_map: bool = '--static-map' in sys.argv
        self.physics_steps_per_observation: int = physics_steps_per_observation

        # For faster / slower than real-time stepping
        self.decouple_step_time = decouple_step_time

        self.episode_steps: int = 0
        self.num_episodes: int = 0
        self.total_steps: int = 0
        self.last_step_time: float = None
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
        self.map_query_seconds_ahead: np.array = np.array([0.5, 1, 1.5, 2, 2.5, 3])
        self.fps: int = 60
        self.target_dt: float = 1 / self.fps
        self.total_episode_time: float = 0
        self.distance: float = 0
        self.distance_to_end: float = 0
        self.prev_distance: float = 0
        self.furthest_distance: float = 0
        self.velocity: List[float] = [0, 0]
        self.angular_velocity: float = 0
        self.gforce: float = 0
        self.gforce_levels: Box = self.blank_gforce_levels()
        self.max_gforce: float = 0
        self.closest_map_index: int = 0
        self.trip_pct: float = 0
        self.avg_trip_pct: float = 0
        self._trip_pct_total: float = 0
        self.angles_ahead: List[float] = []
        self.angle_accuracies: List[float] = []
        self.episode_gforces: List[float] = []
        self.match_angle_only: bool = match_angle_only
        self.one_waypoint_map: bool = one_waypoint_map
        self.incent_win: bool = incent_win
        self.gamma: float = gamma
        self.add_static_obstacle: bool = add_static_obstacle
        self.static_obstacle_points: np.array = None
        self.static_obst_pixels: np.array = None
        self.static_obstacle_tuple: tuple = ()

        # 0.22 m/s on 0.1
        self.max_one_waypoint_mult = 0.5  # Less than 2.5 m/s on 0.1?

        if '--no-timeout' in sys.argv:
            max_seconds = 100000
        elif one_waypoint_map in sys.argv:
            self.one_waypoint_map = True
            max_seconds = self.max_one_waypoint_mult * 200
        else:
            max_seconds = 60
        self._max_episode_steps = \
            max_seconds * 1/self.target_dt * 1/self.physics_steps_per_observation

        self.acceleration = [0, 0]
        self.vehicle_model = None

        # Current position (center)
        self.x = None
        self.y = None
        self.ego_pos = None

        # Angle in radians, 0 is straight up, -pi/2 is right
        self.angle = None

        # Start position
        self.start_x = None
        self.start_y = None
        self.start_angle = None

        # Map
        self.map = None
        self.map_kd_tree = None
        self.map_flat = None

        # Take last n (10 from 0.5 seconds) state, action, reward values and append them
        # to the observation. Should change frame rate?
        self.experience_buffer = None
        self.should_add_previous_states = '--disable-prev-states' not in sys.argv
        np.random.seed(self.seed_value)

        # Actions per second
        self.aps = self.fps / self.physics_steps_per_observation

        self.player = None
        self.ego_rect: np.array = None

        self.reset()

    def enable_render(self):
        from deepdrive_2d import player
        self.player = player.start(
            env=self,
            fps=self.fps / self.physics_steps_per_observation)
        pyglet.app.event_loop.has_exit = False
        pyglet.app.event_loop._legacy_setup()
        pyglet.app.platform_event_loop.start()
        pyglet.app.event_loop.dispatch_event('on_enter')
        pyglet.app.event_loop.is_running = True

        self.should_render = True

    def populate_observation(self, closest_map_point, lane_deviation,
                             angles_ahead, steer, brake, accel, harmful_gs,
                             jarring_gs, uncomfortable_gs, is_blank=False):
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
                cog_to_rear_axle=self.vehicle_model[1],)


        if self.return_observation_as_array:
            # TODO: Remove multi-waypoint stuff as all driving can be
            #  simplified to reaching single waypoint with desired speed and
            #  heading (right?!?). Static and dynamic obstacles can interfere with
            #  ability to reach waypoint, but skipping waypoints should not
            #  be an immediate alternative. Training can then focus on minimizing
            #  g-forces if we don't reach the waypoint. Then at test time
            #  we can set a new waypoint after some timeout.
            if self.one_waypoint_map:
                if self.match_angle_only:
                    return np.array([angles_ahead[0], self.prev_steer])
                elif 'STRAIGHT_TEST' in os.environ:
                    return np.array([self.speed])
                else:
                    ret = [angles_ahead[0], self.prev_steer, self.prev_accel,
                           self.speed, self.distance_to_end]
                    if self.add_static_obstacle:
                        ret = self.add_static_obstacle_inputs(ret)
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
                            observation, self.total_episode_time)
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

        return observation

    def add_static_obstacle_inputs(self, ret):
        start_static_obs = self.static_obstacle_points[0]
        end_static_obs = self.static_obstacle_points[1]
        start_obst_angle = self.get_angle_to_point(
            start_static_obs)
        end_obst_angle = self.get_angle_to_point(
            self.static_obstacle_points[1])
        start_obst_dist = np.linalg.norm(
            start_static_obs - self.ego_pos)
        end_obst_dist = np.linalg.norm(
            end_static_obs - self.ego_pos)
        ret += [start_obst_dist, end_obst_dist,
                start_obst_angle, end_obst_angle]
        return ret

    def reset(self):
        self.episode_steps = 0
        self.angle = self.start_angle
        self.angle_change = 0
        self.x = self.start_x
        self.y = self.start_y
        self.ego_pos = np.array((self.x, self.y))
        self.angle_change = 0
        self.speed = 0
        self.episode_reward = 0
        self.total_episode_time = 0
        self.distance = 0
        self.prev_distance = 0
        self.furthest_distance = 0
        self.velocity = [0, 0]
        self.angular_velocity = 0
        self.acceleration = [0, 0]
        self.gforce = 0
        self.gforce_levels = self.blank_gforce_levels()
        self.max_gforce = 0
        self.closest_map_index = 0
        self.trip_pct = 0
        self.angles_ahead = []
        self.angle_accuracies = []
        self.episode_gforces = []
        # TODO: Regen map every so often
        if self.map is None or not self.static_map:
            self.generate_map()
        if self.observation_space is None:
            self.setup_spaces()
        self.experience_buffer.reset()
        obz = self.get_blank_observation()
        return obz

    def setup_spaces(self):
        # Action space: ----
        # Accel, Brake, Steer
        if self.expect_normalized_actions:
            self.action_space = spaces.Box(low=-1, high=1,
                                           shape=(self.num_actions,))
        else:
            # https://www.convert-me.com/en/convert/acceleration/ssixtymph_1.html?u=ssixtymph_1&v=7.4
            # Max voyage accel m/s/f = 3.625 * FPS = 217.5 m/s/f
            # TODO: Set steering limits as well
            self.action_space = spaces.Box(low=-10.2, high=10.2,
                                           shape=(self.num_actions,))
        self.experience_buffer = ExperienceBuffer()
        blank_obz = self.get_blank_observation()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(blank_obz),))

    @staticmethod
    def blank_gforce_levels():
        return Box(harmful=False, jarring=False,
                   uncomfortable=False)

    def get_blank_observation(self):
        ret = self.populate_observation(
            steer=0,
            brake=0,
            accel=0,
            closest_map_point=self.map.arr[0],
            lane_deviation=0,
            angles_ahead=[0] * len(self.map_query_seconds_ahead),
            harmful_gs=False,
            jarring_gs=False,
            uncomfortable_gs=False,
            is_blank=True,)
        return ret

    def generate_map(self):
        static_obst_pixels = None
        # Generate one waypoint map
        if self.one_waypoint_map:
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


        else:
            x, y = gen_map(should_save=True)

        x_pixels = x * MAP_WIDTH_PX + SCREEN_MARGIN
        y_pixels = y * MAP_HEIGHT_PX + SCREEN_MARGIN


        x_meters = x_pixels / self.px_per_m
        y_meters = y_pixels / self.px_per_m

        arr = list(zip(list(x_meters), list(y_meters)))

        distances = np.cumsum(np.linalg.norm(np.diff(arr, axis=0), axis=1))
        distances = np.concatenate((np.array([0]), distances))

        self.map = Box(x=x_meters,
                       y=y_meters,
                       x_pixels=x_pixels,
                       y_pixels=y_pixels,
                       arr=arr,
                       distances=distances,
                       length=distances[-1],
                       width=(MAP_WIDTH_PX + SCREEN_MARGIN) / self.px_per_m,
                       height=(MAP_HEIGHT_PX + SCREEN_MARGIN) / self.px_per_m,
                       static_obst_pixels=self.static_obst_pixels)

        self.x = self.map.x[0]
        self.y = self.map.y[0]
        self.ego_pos = np.array((self.x, self.y))

        self.start_x = self.x
        self.start_y = self.y

        # Physics properties
        # x is right, y is straight
        self.map_kd_tree = spatial.KDTree(self.map.arr)
        self.vehicle_model = self.get_vehicle_model(self.vehicle_width)

        self.map_flat = flatten_points(self.map.arr)
        if self.one_waypoint_map:
            self.angle = -math.pi / 2
        else:
            self.angle = self.get_start_angle()
        self.start_angle = self.angle

    def seed(self, seed=None):
        self.seed_value = seed or 0
        random.seed(seed)

    def get_start_angle(self):
        interp_dist_pixels = self.map.width / len(self.map.x)
        angle_waypoint_meters = 1
        # angle_waypoint_index = round(
        #     (angle_waypoint_meters * ROUGH_PIXELS_PER_METER) /
        #     interp_dist_pixels) + 1
        angle_waypoint_index = min(6, len(self.map.x) - 1)
        x1 = self.map.x[0]
        y1 = self.map.y[0]
        x2 = self.map.x[angle_waypoint_index]
        y2 = self.map.y[angle_waypoint_index]
        # self.heading_x = x2
        # self.heading_y = y2
        angle = angle_between_points([x1, y1], [x2, y2])
        if (self.map.height - y1) / self.map.height <= 0.5 and \
                abs(angle) < (math.pi * 0.001):
            # TODO: Reproduce problem where car is backwards along spline
            #  and make sure this fixes it.
            log.warning('Flipped car to avoid being upside down. Did it work?')
            angle += math.pi  # On top so, face down
        log.info(f'Start angle is {angle}')

        return angle

    @staticmethod
    def get_vehicle_model(width):
        # Bias towards the front a bit
        # https://www.fcausfleet.com/content/dam/fca-fleet/na/fleet/en_us/chrysler/2017/pacifica/vlp/docs/Pacifica_Specifications.pdf
        if USE_VOYAGE:
            bias_towards_front = .05 * width
        else:
            bias_towards_front = 0

        # Center of gravity
        center_of_gravity = (width / 2) + bias_towards_front
        # Approximate axles to be 1/8 (1/4 - 1/8) from ends of car
        rear_axle = width * 1 / 8
        front_axle = width - rear_axle
        L_b = center_of_gravity - rear_axle
        L_a = front_axle - center_of_gravity
        return L_a, L_b

    @log.catch
    def step(self, action):
        if IS_DEBUG_MODE:
            return self._step(action)
        else:
            # Fail gracefully when running so that long training runs are
            # not interrupted by transient errors
            try:
                return self._step(action)
            except:
                log.exception('Caught exception in step, ending episode')
                obz = self.get_blank_observation()
                done = True
                if '--penalize-loss' in sys.argv:
                    reward = GAME_OVER_PENALTY
                else:
                    reward = 0
                info = {}

                return obz, reward, done, info

    @log.catch
    def _step(self, action):
        # Retry added to deal with weird kd_tree exception where index was beyond
        # length of 135 when length should have been 175
        dt = self.get_dt()
        info = Box(default_box=True)
        if 'STRAIGHT_TEST' in os.environ:
            accel = action[0]
            steer = 0
            brake = 0
        else:
            steer, accel, brake = action

        steer, accel, brake = self.denormalize_actions(steer, accel, brake)

        if 'STRAIGHT_TEST' in os.environ:
            steer = 0
            brake = 0
            # accel = MAX_METERS_PER_SEC_SQ

        if '--straight-test' in sys.argv:
            steer = 0
        elif '--simple-steer' in sys.argv and self.angles_ahead:
            accel = MAX_METERS_PER_SEC_SQ * 0.7
            steer = -self.angles_ahead[0]
        elif self.match_angle_only:
            accel = MAX_METERS_PER_SEC_SQ * 0.7

        info.stats.steer = steer
        info.stats.accel = accel
        info.stats.brake = brake
        info.stats.speed = self.speed
        info.stats.episode_time = self.total_episode_time

        now = time.time()
        if self.last_step_time is None:
            # init
            self.last_step_time = now
            reward = 0
            done = False
            observation = self.get_blank_observation()
        else:
            collided = self.check_for_collisions()

            lane_deviation, observation, closest_map_point = \
                self.get_observation(steer, accel, brake, dt, info)
            done, won, lost = self.get_done(closest_map_point, lane_deviation,
                                            collided)
            reward, info = self.get_reward(lane_deviation, won, lost, collided,
                                           info, steer, accel)
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
            log.debug(f'Episode score {round(self.episode_reward, 2)}, '
                      f'Steps: {self.episode_steps}, '
                      # f'Closest map indx: {self.closest_map_index}, '
                      f'Distance {round(self.distance, 2)}, '
                      f'Angular velocity {round(self.angular_velocity, 2)}, '
                      f'Speed: {round(self.speed, 2)}, '
                      f'Max gforce: {round(self.max_gforce, 4)}, '
                      f'Avg gforce: {round(episode_gforce_avg, 4)}, '
                      f'Trip pct {round(self.trip_pct, 2)}, '
                      f'Angle accuracy {round(episode_angle_accuracy, 2)}, '
                      f'Num episodes {self.num_episodes}')

        self.total_steps += 1
        self.prev_steer = steer
        self.prev_accel = accel
        self.prev_brake = brake

        self.ego_rect, self.ego_rect_tuple = get_rect(
            self.x, self.y, self.angle, self.vehicle_width, self.vehicle_height)

        self.ego_pos = np.array((self.x, self.y))
        self.front_x = self.x + cos(pi/2 + self.angle) * self.vehicle_height / 2
        self.front_y = self.y + sin(pi/2 + self.angle) * self.vehicle_width / 2

        return observation, reward, done, info.to_dict()

    def denormalize_actions(self, steer, accel, brake):
        # TODO: Numba this
        if self.expect_normalized_actions:
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

            steer = steer * pi / 6  # About 33 degrees max steer

            # Forward only for now
            accel = MAX_METERS_PER_SEC_SQ * ((1 + accel) / 2)


            brake = MAX_METERS_PER_SEC_SQ * ((1 + brake) / 2)
        return steer, accel, brake

    def get_dt(self):
        if self.decouple_step_time:
            dt = self.target_dt
        else:
            dt = time.time() - self.last_step_time
        return dt

    def get_done(self, closest_map_point, lane_deviation,
                 collided: bool) -> Tuple[bool, bool, bool]:
        done = False
        won = False
        lost = False
        if collided:
            log.warning(f'Collision, game over.')
            done = True
            lost = True
        if lane_deviation > 1.1 and not self.one_waypoint_map:
            # You lose!
            log.warning(f'Drifted out of lane, game over.')
            done = True
            lost = True
        elif self.gforce_levels.harmful and \
                self.should_penalize_gforce(min_trip_complete=0.25):
            # Only end on g-force once we've learned to complete part of the trip.
            log.warning(f'Harmful g-forces, game over')
            done = True
            lost = True
        elif (self.episode_steps + 1) % self._max_episode_steps == 0:
            log.warning(f'Time up')
            done = True
        elif self.one_waypoint_map:
            if abs(math.degrees(self.angle)) > 200:
                done = True
                lost = True
                log.warning(f'Going in circles - angle {math.degrees(self.angle)} too high')
            if (self.furthest_distance - self.distance) > 2:
                done = True
                lost = True
                log.warning(f'Negative progress')
            elif abs(self.map.length - self.distance) < 1:
                done = True
                won = True
                # You win!
                log.success(f'Reached destination! '
                            f'Steps: {self.episode_steps}')
        elif list(self.map.arr[-1]) == list(closest_map_point):
            # You win!
            log.success(f'Reached destination! '
                        f'Steps: {self.episode_steps}')
            done = True
            won = True
        if '--test-win' in sys.argv:
            won = True
        return done, won, lost


    def get_reward(self, lane_deviation: float,  won: bool, lost: bool,
                   collided: bool, info: Box, accel: float) -> Tuple[float, Box]:
        reward = 0
        target_mps = 15

        if self.one_waypoint_map:

        angle_diff = abs(self.angles_ahead[0])

        if 'STRAIGHT_TEST' in os.environ:
            angle_reward = 0
        else:
            angle_reward = 4 * pi - angle_diff

        angle_accuracy = 1 - angle_diff / (2 * pi)

            # Add speed reward (TODO: Make this same as minimizing trip time)
            # if ret > 0:
            #     if self.speed == 0:
            #         ret = 0
            #     else:
            #         ret += self.speed * 8 * pi
            frame_distance = self.distance - self.furthest_distance
            speed_reward = 0
            if frame_distance > 0:
                # With distance:
                # 32: Speed 1.54, max-g: 0.5, 16: speed 1, max-g: 0.1
                # 8: Speed 0.1, max-g: 0.03
                # Waypoint mult 0.5: 8: Speed 0.3, avg-g: 0.01, max-g: 0.1
                #
                # With speed * 8 * pi: Speed 3.8, max-g: 0.71
                speed_reward = frame_distance * 8 * pi
                self.furthest_distance = self.distance

        if 'ACTION_PENALTY' in os.environ:
            action_penalty = float(os.environ['ACTION_PENALTY'])
            steer_penalty = steer * action_penalty
            accel_penalty = accel * action_penalty
        else:
            steer_penalty = 0
            accel_penalty = 0

            gforce_reward = 0
            if self.gforce > 0.05:
                gforce_reward = -8 * pi * self.gforce  # G-force penalty
            self.angle_accuracies.append(angle_accuracy)
            info.stats.angle_accuracy = angle_accuracy

        if collided:
            collision_penalty = pi
        else:
            collision_penalty = 0

        ret = (
           + speed_reward
           - gforce_penalty
           + self.get_win_reward(won)
           - collision_penalty
           - steer_penalty
           - accel_penalty
        )


            # ret = gforce_reward
            # ret = angle_reward + speed_reward
            # ret = self.speed
            # ret = self.gforce

        # log.trace(f'reward {ret} '
        #          f'angle {angle_reward} '
        #          f'speed {speed_reward} '
        #          f'gforce {gforce_reward}')
        return ret, info

        if self.gforce_levels.jarring and self.should_penalize_gforce():
            # log.warning(f'Jarring g-forces')
            reward = -1
        elif self.gforce_levels.uncomfortable and self.should_penalize_gforce():
            # log.warning(f'Uncomfortable g-forces')
            reward = -0.5
        elif self.distance > self.furthest_distance and \
                '--award-win-only' not in sys.argv:
            target_per_frame_distance = target_mps / self.aps
            frame_distance = self.distance - self.furthest_distance
            if 0 <= frame_distance <= target_per_frame_distance:
                # Positive progress made under speed limit
                reward = frame_distance / target_per_frame_distance
            elif frame_distance > target_per_frame_distance:
                # Over desired speed
                reward = 0
            else:
                # Negative progress
                reward = -0.5
            self.furthest_distance = self.distance
        else:
            reward = 0

        if '--penalize-loss' in sys.argv and lost:
            reward = GAME_OVER_PENALTY

        reward = self.get_win_reward(won) or reward

        return reward, info

    def get_win_reward(self, won):
        win_reward = 0
        if self.incent_win and won:
            win_reward = self.map.length * 8 * pi
        return win_reward

    def get_observation(self, steer, accel, brake, dt, info):
        if self.ignore_brake:
            brake = False
        if self.speed > 100:
            log.warning('Cutting off throttle at speed > 100m/s')
            accel = 0

        prev_x, prev_y, prev_angle, prev_angle_change = \
            self.x, self.y, self.angle, self.angle_change

        self.step_physics(dt, steer, accel, brake, prev_x, prev_y, prev_angle,
                          info)

        closest_map_point, closest_map_index, lane_deviation = \
            get_closest_point((self.x, self.y), self.map_kd_tree)

        self.closest_map_index = closest_map_index
        self.set_distance(closest_map_index)

        if self.one_waypoint_map:
            angles_ahead = self.get_one_waypoint_angle_ahead()
            self.trip_pct = 100 * self.distance / self.map.length
            # log.info(f'angle ahead {math.degrees(angles_ahead[0])}')
            # log.info(f'angle {math.degrees(self.angle)}')
        else:
            self.trip_pct = 100 * closest_map_index / (len(self.map.arr) - 1)
            angles_ahead = self.get_angles_ahead(closest_map_index)

        self.angles_ahead = angles_ahead

        info.stats.closest_map_index = closest_map_index
        info.stats.trip_pct = self.trip_pct
        info.stats.distance = self.distance

        observation = self.populate_observation(
            closest_map_point=closest_map_point,
            lane_deviation=lane_deviation,
            angles_ahead=angles_ahead,
            steer=steer,
            brake=brake,
            accel=accel,
            harmful_gs=self.gforce_levels.harmful,
            jarring_gs=self.gforce_levels.jarring,
            uncomfortable_gs=self.gforce_levels.uncomfortable)

        return lane_deviation, observation, closest_map_point

    def get_one_waypoint_angle_ahead(self):
        angle_to_waypoint = self.get_angle_to_point([self.map.x[-1],
                                                    self.map.y[-1]])

        # Repeat to match dimensions of many waypoint map
        ret = [angle_to_waypoint] * len(self.map_query_seconds_ahead)

        return ret

    def get_angle_to_point(self, p):
        angle_to_dest = angle_between_points([self.x, self.y], [p[0], p[1]])
        angle_ahead = self.angle - angle_to_dest
        return angle_ahead

    def step_physics(self, dt, steer, accel, brake, prev_x, prev_y, prev_angle,
                     info):
        n = self.physics_steps_per_observation
        self.gforce_levels = self.blank_gforce_levels()
        # TODO: Numba this
        for i in range(n):
            interp = (i + 1) / n
            i_steer = self.prev_steer + interp * (steer - self.prev_steer)
            i_accel = self.prev_accel + interp * (accel - self.prev_accel)
            if brake:
                i_brake = self.prev_brake + interp * (float(brake) - self.prev_brake)
            else:
                i_brake = 0
            # log.info(f'steer {steer} accel {accel} brake {brake} vel {self.velocity}')
            prev_x, prev_y, prev_angle = self.x, self.y, self.angle
            self.x, self.y, self.angle, self.angle_change, self.speed = \
                bike_with_friction_step(
                    steer=i_steer, accel=i_accel, brake=i_brake, dt=dt,
                    x=self.x, y=self.y, angle=self.angle,
                    angle_change=self.angle_change,
                    speed=self.speed,
                    add_rotational_friction=self.add_rotational_friction,
                    add_longitudinal_friction=self.add_longitudinal_friction,
                    vehicle_model=self.vehicle_model,)
            self.check_for_nan(dt, i_steer, i_accel, i_brake, info)

            self.get_gforce_levels(dt, prev_angle, prev_x, prev_y, info)

        self.episode_gforces.append(self.gforce)

    def render(self, mode='human'):
        platform_event_loop = pyglet.app.platform_event_loop
        # pyglet_event_loop = pyglet.app.event_loop
        timeout = pyglet.app.event_loop.idle()
        platform_event_loop.step(timeout)

    def set_distance(self, closest_map_index):
        self.prev_distance = self.distance
        if 'STRAIGHT_TEST' in os.environ:
            self.distance = self.x - self.start_x
        elif self.one_waypoint_map:
            end = np.array([self.map.x[-1], self.map.y[-1]])
            pos = np.array([self.front_x, self.front_y])
            self.distance_to_end = np.linalg.norm(end - pos)
            self.distance = self.map.length - self.distance_to_end
            # log.info(f'distance {self.distance}')
        else:
            self.distance = self.map.distances[closest_map_index]
        self.furthest_distance = max(self.distance, self.furthest_distance)
        if self.prev_distance is None:
            # Init prev distance
            self.prev_distance = self.distance

    def get_gforce_levels(self, dt, prev_angle, prev_x, prev_y, info):
        # TODO: Numba this
        lvls = self.gforce_levels
        self.total_episode_time += dt
        pos_change = np.array([prev_x - self.x, prev_y - self.y])
        prev_velocity = self.velocity
        self.velocity = pos_change / dt
        self.acceleration = (self.velocity - prev_velocity) / dt
        self.angular_velocity = (self.angle - prev_angle) / dt
        accel_magnitude = np.linalg.norm(self.acceleration)
        gforce = accel_magnitude / 9.807
        if gforce > self.max_gforce:
            log.trace(f'New max gforce {gforce}')
        self.max_gforce = max(gforce, self.max_gforce)
        info.stats.gforce = gforce
        if gforce > 1:
            lvls.harmful = True
            info.stats.episode_gforce_level = 3
            log.trace(f'Harmful gforce encountered {gforce} '
                      f'speed: {self.speed}')
        elif gforce > 0.4 and not lvls.harmful:
            lvls.jarring = True
            info.stats.episode_gforce_level = 2
            log.trace(f'Jarring gforce encountered {gforce} '
                        f'speed: {self.speed}')
        elif gforce > 0.1 and not (lvls.harmful or lvls.jarring):
            lvls.uncomfortable = True
            info.stats.episode_gforce_level = 1
            log.trace(f'Uncomfortable gforce encountered {gforce} '
                      f'speed: {self.speed}')
        elif not (lvls.harmful or lvls.jarring or lvls.uncomfortable):
            info.stats.episode_gforce_level = 0
        self.gforce = gforce


    def check_for_nan(self, dt, steer, accel, brake, info):
        if self.x in [np.inf, -np.inf] or self.y in [np.inf, -np.inf]:
            # Something has went awry in our bike model
            log.error(f"""
Position is infinity 
self.x, self.y, self.angle, self.angle_change, self.speed
{self.x, self.y, self.angle, self.angle_change, self.speed}

steer, accel, brake, dt, info
{steer, accel, brake, dt, info}
""")
            raise RuntimeError('Position is infinity')

    def get_angles_ahead(self, closest_map_index):
        """
        Note: this assumes we are on the map path
        """
        distances = self.map.distances
        return get_angles_ahead(total_points=len(distances),
                                total_length=self.map.length,
                                speed=self.speed,
                                map_points=self.map.arr,
                                angle=self.angle,
                                seconds_ahead=self.map_query_seconds_ahead,
                                closest_map_index=closest_map_index)

    def close(self):
        if self.should_render:
            pyglet.app.is_running = False
            pyglet.app.dispatch_event('on_exit')
            pyglet.app.platform_event_loop.stop()

    def check_for_collisions(self):
        if 'DISABLE_COLLISION_CHECK' in os.environ:
            return False
        elif self.add_static_obstacle:
            return check_collision(self.ego_rect_tuple,
                                   lines=(self.static_obstacle_tuple,))


@njit(cache=True, nogil=True)
def bike_with_friction_step(
        steer, accel, brake, dt,
        x, y, angle, angle_change, speed, add_rotational_friction,
        add_longitudinal_friction, vehicle_model):
    steer = min(pi, steer)
    steer = max(-pi, steer)
    state = [x, y, angle_change, speed]

    change_x, change_y, angle_change, speed = \
        f_KinBkMdl(state, steer, accel, vehicle_model, dt)

    tuned_fps = 1 / 60  # The FPS we tuned friction ratios at
    friction_exponent = (dt / tuned_fps)
    if add_rotational_friction:
        angle_change = 0.95 ** friction_exponent * angle_change
    if add_longitudinal_friction:
        speed = 0.999 ** friction_exponent * speed
    if brake:
        speed = 0.97 ** friction_exponent * speed
        # self.speed = 0.97 * self.speed

    theta1 = angle
    theta2 = theta1 + pi / 2
    world_change_x = change_x * cos(theta2) + change_y * cos(theta1)
    world_change_y = change_x * sin(theta2) + change_y * sin(theta1)

    x += world_change_x
    y += world_change_y

    # self.player_sprite.center_x += world_change_x
    # self.player_sprite.center_y += world_change_y
    # self.player_sprite.angle += degrees(self.yaw_rate)
    angle += angle_change

    return x, y, angle, angle_change, speed


@njit(cache=True, nogil=True)
def f_KinBkMdl(state, steer_angle, accel, vehicle_model, dt):
    """
    process model
    input: state at time k, z[k] := [x[k], y[k], psi[k], v[k]]
    output: state at next time step z[k+1]
    # From https://github.com/MPC-Berkeley/barc/blob/master/workspace/src/barc/src/estimation/system_models.py
    """

    # get states / inputs
    x = state[0]             # straight
    y = state[1]             # right
    angle_change = state[2]  # angle change
    speed = state[3]         # speed

    # extract parameters
    # Distance from center of gravity to front and rear axles
    (L_a, L_b) = vehicle_model

    # compute slip angle
    slip_angle = np.arctan(L_a / (L_a + L_b) * np.tan(steer_angle))

    # compute next state
    change_x = dt * (speed * np.cos(angle_change + slip_angle))
    change_y = dt * (speed * np.sin(angle_change + slip_angle))
    angle_change = angle_change + dt * speed / L_b * np.sin(slip_angle)
    speed_next = speed + dt * accel

    return np.array([change_x, change_y, angle_change, speed_next])


def get_closest_point(point, kd_tree):
    distance, index = kd_tree.query(point)
    if index >= kd_tree.n:
        log.warning(f'kd tree index out of range, using last index. '
                    f'point {point}\n'
                    f'kd_tree: {json.dumps(kd_tree.data.tolist(), indent=2)}')
        index = kd_tree.n - 1
    point = kd_tree.data[index]
    return point, index, distance


def np_rand():
    return np.random.rand(1)[0]


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



def main():
    env = Deepdrive2DEnv()



if __name__ == '__main__':
    if '--test_static_obstacle' in sys.argv:
        test_static_obstacle()
    elif '--test_get_rect' in sys.argv:
        test_get_rect()
    else:
        main()
