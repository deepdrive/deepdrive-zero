import math
import os
import sys
import time
from collections import deque
from math import pi, cos, sin

import gym
import numpy as np
from box import Box
from gym import spaces
from loguru import logger as log
from numba import njit
from scipy import spatial


from deepdrive_2d.constants import USE_VOYAGE, MAP_WIDTH_PX, MAP_HEIGHT_PX, \
    SCREEN_MARGIN, VEHICLE_HEIGHT, VEHICLE_WIDTH, PX_PER_M, \
    MAX_METERS_PER_SEC_SQ
from deepdrive_2d.experience_buffer import ExperienceBuffer
from deepdrive_2d.map_gen import gen_map
from deepdrive_2d.utils import flatten_points, get_angles_ahead, \
    get_heading

MAX_BRAKE_G = 1
G_ACCEL = 9.80665
CONTINUOUS_REWARD = True


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
                 static_map=True):

        # All units in meters and radians unless otherwise specified
        self.vehicle_width = vehicle_width
        self.vehicle_height = vehicle_height
        self.return_observation_as_array = return_observation_as_array
        self.px_per_m = px_per_m
        self.ignore_brake = ignore_brake
        self.expect_normalized_actions = expect_normalized_actions
        self.seed_value = seed_value
        self.add_rotational_friction = add_rotational_friction
        self.add_longitudinal_friction = add_longitudinal_friction
        self.static_map = static_map

        # For faster / slower than real-time stepping
        self.decouple_step_time = decouple_step_time

        self.step_num: int = 0
        self.last_step_time = None
        self._max_episode_steps = 10 ** 3
        self.num_actions = 3  # Steer, throttle, brake
        self.prev_action = [0] * self.num_actions
        self.episode_reward = 0
        self.speed = 0
        self.angle_change = 0
        self.map_query_seconds_ahead = np.array([0.5, 1, 1.5, 2, 2.5, 3])
        self.target_dt = 1 / 60  # Allows replaying in player at same rate
        self.total_episode_time = 0
        self.episode_distance = 0
        self.prev_episode_distance = 0


        self.map = None
        self.x = None
        self.y = None
        self.start_x = None
        self.start_y = None
        self.map_kd_tree = None
        self.vehicle_model = None
        self.angle = None
        self.start_angle = None
        self.map_flat = None

        # Take last n (10 from 0.5 seconds) state, action, reward values and append them
        # to the observation. Should change frame rate?
        self.experience_buffer = None
        self.should_add_previous_states = '--disable-prev-states' not in sys.argv


    def setup(self):
        np.random.seed(self.seed_value)

        self.map = self.generate_map()

        self.x = self.map.x[0]
        self.y = self.map.y[0]

        self.start_x = self.x
        self.start_y = self.y

        # Physics properties
        # x is right, y is straight
        self.map_kd_tree = spatial.KDTree(self.map.arr)
        self.vehicle_model = self.get_vehicle_model(self.vehicle_width)

        self.map_flat = flatten_points(self.map.arr)
        self.angle = self.get_start_angle()
        self.start_angle = self.angle

        # Action space: ----
        # Accel, Brake, Steer
        if self.expect_normalized_actions:
            self.action_space = spaces.Box(low=-1, high=1,
                                           shape=(self.num_actions,))
        else:
            # https://www.convert-me.com/en/convert/acceleration/ssixtymph_1.html?u=ssixtymph_1&v=7.4
            # Max voyage accel m/s/f = 3.625 * FPS = 217.5 m/s/f
            self.action_space = spaces.Box(low=-10.2, high=10.2,
                                           shape=(self.num_actions,))

        self.experience_buffer = ExperienceBuffer()
        blank_obz = self.get_blank_observation()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(blank_obz),))
        return blank_obz

    def populate_observation(self, closest_map_point, lane_deviation,
                             angles_ahead, steer, brake, accel, is_blank=False):
        observation = \
            Box(steer=steer,
                accel=accel,
                brake=brake,
                x=self.x,
                y=self.y,
                angle=self.angle,
                speed=self.speed,
                distance=self.episode_distance,
                angle_change=self.angle_change,
                closest_map_point_x=closest_map_point[0],
                closest_map_point_y=closest_map_point[1],
                lane_deviation=lane_deviation,
                vehicle_width=self.vehicle_width,
                vehicle_height=self.vehicle_height,
                cog_to_front_axle=self.vehicle_model[0],
                cog_to_rear_axle=self.vehicle_model[1],)
        if self.return_observation_as_array:
            observation = np.array(observation.values())
            observation = np.concatenate((observation, angles_ahead), axis=None)

            if self.should_add_previous_states:
                if self.experience_buffer.size() == 0:
                    self.experience_buffer.setup(shape=(len(observation),))
                if is_blank:
                    past_values = np.concatenate(np.array(
                        self.experience_buffer.blank_buffer), axis=0)
                else:
                    self.experience_buffer.maybe_add(observation,
                                                     self.total_episode_time)
                    past_values = np.concatenate(np.array(
                        self.experience_buffer.buffer), axis=0)
                observation = np.concatenate((observation, past_values),
                                             axis=None)
        else:
            observation.angles_ahead = angles_ahead

        return observation

    def reset(self):

        self.step_num = 0
        self.angle = self.start_angle
        self.angle_change = 0
        self.x = self.start_x
        self.y = self.start_y
        self.angle_change = 0
        self.speed = 0
        self.episode_reward = 0
        self.total_episode_time = 0
        self.episode_distance = 0
        self.prev_episode_distance = 0

        # TODO: Regen map every so often
        if self.map is None:
            obz = self.setup()
        else:
            self.experience_buffer.reset()
            obz = self.get_blank_observation()

        if self.static_map:
            np.random.seed(self.seed_value)

        return obz

    def get_blank_observation(self):
        ret = self.populate_observation(
            steer=0,
            brake=0,
            accel=0,
            closest_map_point=self.map.arr[0],
            lane_deviation=0,
            angles_ahead=[0] * len(self.map_query_seconds_ahead),
            is_blank=True,
        )
        return ret

    def generate_map(self):
        x, y = gen_map(should_save=True)

        x_pixels = x * MAP_WIDTH_PX + SCREEN_MARGIN
        y_pixels = y * MAP_HEIGHT_PX + SCREEN_MARGIN

        x_meters = x_pixels / self.px_per_m
        y_meters = y_pixels / self.px_per_m

        arr = list(zip(list(x_meters), list(y_meters)))

        distances = np.cumsum(np.linalg.norm(np.diff(arr, axis=0), axis=1))
        distances = np.concatenate((np.array([0]), distances))

        ret = Box(x=x_meters,
                  y=y_meters,
                  x_pixels=x_pixels,
                  y_pixels=y_pixels,
                  arr=arr,
                  distances=distances,
                  length=distances[-1],
                  width=(MAP_WIDTH_PX + SCREEN_MARGIN) / self.px_per_m,
                  height=(MAP_HEIGHT_PX + SCREEN_MARGIN) / self.px_per_m, )
        return ret

    def seed(self, seed=None):
        self.seed_value = seed or 0

    def get_start_angle(self):
        interp_dist_pixels = self.map.width / len(self.map.x)
        angle_waypoint_meters = 1
        # angle_waypoint_index = round(
        #     (angle_waypoint_meters * ROUGH_PIXELS_PER_METER) /
        #     interp_dist_pixels) + 1
        angle_waypoint_index = 6
        x1 = self.map.x[0]
        y1 = self.map.y[0]
        x2 = self.map.x[angle_waypoint_index]
        y2 = self.map.y[angle_waypoint_index]
        # self.heading_x = x2
        # self.heading_y = y2
        angle = get_heading([x1, y1], [x2, y2])
        if (self.map.height - y1) / self.map.height <= 0.5 and \
                abs(angle) < (math.pi * 0.001):
            # TODO: Reproduce problem where car is backwards along spline
            #  and make sure this fixes it.
            log.warning('Flipped car to avoid being upside down. Did it work?')
            angle += math.pi  # On top so, face down
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

    def step(self, action):
        info = Box(default_box=True)
        steer, accel, brake = action
        accel, steer = self.normalize_actions(accel, steer)
        info.tfx.steer = steer
        info.tfx.accel = accel
        info.tfx.brake = brake
        info.tfx.speed = self.speed

        if self.last_step_time is None:
            # init
            self.last_step_time = time.time()
            reward = 0
            done = False
            observation = self.get_blank_observation()
        else:
            dt = self.get_dt()
            # TODO: Car lists with collision detection
            lane_deviation, observation, closest_map_point = \
                self.get_observation(steer, accel, brake, dt, info)
            reward = self.get_reward(lane_deviation)
            done = self.get_done(closest_map_point, lane_deviation, info)
            info.tfx.lane_deviation = lane_deviation


        self.last_step_time = time.time()
        self.episode_reward += reward

        self.prev_action = action
        self.step_num += 1
        return observation, reward, done, info.to_dict()

    def normalize_actions(self, accel, steer):
        if self.expect_normalized_actions:
            steer = steer * pi

            # Forward only for now
            accel = MAX_METERS_PER_SEC_SQ * ((1 + accel) / 2)
        return accel, steer

    def get_dt(self):
        if self.decouple_step_time:
            dt = self.target_dt
        else:
            dt = time.time() - self.last_step_time
        self.total_episode_time += dt
        return dt

    def get_done(self, closest_map_point, lane_deviation, info) -> bool:
        done = False

        if lane_deviation > 1.1:
            # You lose!
            log.debug(f'Drifted out of lane, game over. '
                      f'Steps: {self.step_num} '
                      f'Score: {self.episode_reward} '
                      f'Distance {self.episode_distance}')
            done = True
        if list(self.map.arr[-1]) == list(closest_map_point):
            # You win!
            log.success(f'Reached destination! '
                        f'Steps: {self.step_num} Score: {self.episode_reward}')
            done = True
        return done

    def get_reward(self, lane_deviation) -> float:
        if '--distance-only-reward' in sys.argv:
            if self.episode_distance > self.prev_episode_distance:
                reward = self.speed
            else:
                reward = 0
        elif lane_deviation < 1 and self.speed > 2:  # ~5mph
            # TODO: Double check these are meters
            if '--speed-only-reward' in sys.argv:
                reward = self.speed
            elif CONTINUOUS_REWARD:
                max_desired_speed = 8  # ~20mph
                min_desired_speed = 2  # ~5mph
                reward = (self.speed - min_desired_speed) / (
                        max_desired_speed - min_desired_speed)
                reward = min(reward, 1)
                log.debug(f'Reward: {reward}')
            else:
                reward = 1
        else:
            # Avoid negative rewards due to
            # http://bit.ly/mistake_importance_scaling
            reward = 0
        return reward

    def get_observation(self, steer, accel, brake, dt, info):
        if self.ignore_brake:
            brake = False
        self.x, self.y, self.angle, self.angle_change, self.speed = \
            bike_with_friction_step(
                steer=steer, accel=accel, brake=brake, dt=dt,
                x=self.x, y=self.y, angle=self.angle,
                angle_change=self.angle_change,
                speed=self.speed,
                add_rotational_friction=self.add_rotational_friction,
                add_longitudinal_friction=self.add_longitudinal_friction,
                vehicle_model=self.vehicle_model,
            )
        closest_map_point, closest_map_index, lane_deviation = \
            get_closest_point((self.x, self.y), self.map_kd_tree)

        # TODO: Make points ahead distance dependent on speed
        angles_ahead = self.get_angles_ahead(closest_map_index)

        info.tfx.closest_map_index = closest_map_index
        info.tfx.distance = self.map.distances[closest_map_index]

        self.prev_episode_distance = self.episode_distance

        self.episode_distance = self.map.distances[closest_map_index]

        observation = self.populate_observation(closest_map_point=closest_map_point,
                                                lane_deviation=lane_deviation,
                                                angles_ahead=angles_ahead,
                                                steer=steer,
                                                brake=brake,
                                                accel=accel)
        return lane_deviation, observation, closest_map_point

    def get_angles_ahead(self, closest_map_index):
        distances = self.map.distances
        return get_angles_ahead(total_points=len(distances),
                                total_length=self.map.length,
                                speed=self.speed,
                                map_points=self.map.arr,
                                angle=self.angle,
                                seconds_ahead=self.map_query_seconds_ahead,
                                closest_map_index=closest_map_index)

    def render(self, mode='human'):
        pass

    def close(self):
        pass


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
    point = kd_tree.data[index]
    return point, index, distance


def main():
    env = Deepdrive2DEnv()


if __name__ == '__main__':
    main()
