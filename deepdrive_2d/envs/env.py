import math
import time
from math import pi, cos, sin

import numpy as np
from box import Box
from gym import spaces
from loguru import logger as log
from numba import njit
from scipy import spatial


from deepdrive_2d.constants import USE_VOYAGE, MAP_WIDTH_PX, MAP_HEIGHT_PX, \
    SCREEN_MARGIN
from deepdrive_2d.map_gen import gen_map
from deepdrive_2d.utils import angle_between


MAX_BRAKE_G = 1
G_ACCEL = 9.80665
CONTINUOUS_REWARD = True


class Deepdrive2DEnv:
    metadata = {'render.modes': ['human']}

    def __init__(self, vehicle_width, vehicle_height,
                 add_rotational_friction=True,
                 add_longitudinal_friction=True,
                 angle=None,
                 px_per_m=None,
                 return_observation_as_array=True,
                 seed_value=0):

        # All units in meters and radians unless otherwise specified

        self.vehicle_width = vehicle_width
        self.vehicle_height = vehicle_height
        self.angle = angle  # Angle in radians
        self.return_observation_as_array = return_observation_as_array
        self.px_per_m = px_per_m

        self.seed_value = seed_value
        self.map = self.generate_map()

        self.x = self.map.x[0]
        self.y = self.map.y[0]

        # Physics properties
        # x is right, y is straight
        if angle and self.map:
            raise RuntimeError('Cannot specify both angle and map')
        elif self.map:
            self.map_kd_tree = spatial.KDTree(self.map.arr)
        else:
            self.map_kd_tree = None
        self.vehicle_model = self.get_vehicle_model(vehicle_width)

        self.angle_change = 0
        self.speed = 0
        self.add_rotational_friction = add_rotational_friction
        self.add_longitudinal_friction = add_longitudinal_friction
        self.map_flat = [coord for point in self.map.arr for coord in point]
        self.angle = self.get_start_angle(angle)

        # Step properties
        self.step_num: int = 0
        self.last_step_time = None
        self._max_episode_steps = 10 ** 3

        # Action space: ----
        # Accel, Brake, Steer
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))

        # self.observation_space = spaces.Box(low=-1, high=1, shape=(7,))

    def generate_map(self):
        x, y = gen_map(should_save=True)

        x_pixels = x * MAP_WIDTH_PX + SCREEN_MARGIN
        y_pixels = y * MAP_HEIGHT_PX + SCREEN_MARGIN

        x_meters = x_pixels / self.px_per_m
        y_meters = y_pixels / self.px_per_m

        ret = Box(x=x_meters,
                  y=y_meters,
                  x_pixels=x_pixels,
                  y_pixels=y_pixels,
                  arr=list(zip(list(x_meters), list(y_meters))),
                  width=(MAP_WIDTH_PX + SCREEN_MARGIN) / self.px_per_m,
                  height=(MAP_HEIGHT_PX + SCREEN_MARGIN) / self.px_per_m, )
        return ret

    def seed(self, seed):
        # TODO: Use this for map gen
        self.seed_value = seed

    def get_start_angle(self, angle):
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
        angle = -angle_between(np.array([0, 1]), np.array([x2 - x1, y2 - y1]))
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
        steer, accel, brake = action

        observation = self.get_blank_observation()
        done = False
        info = {}
        reward = 0

        if self.last_step_time is None:
            # init
            self.last_step_time = time.time()
        else:
            dt = time.time() - self.last_step_time
            # TODO: Car lists with collision detection
            start = time.time()
            lane_deviation, observation, closest_map_point = \
                self.get_observation(accel, brake, dt, start, steer)
            # TODO: Check to see if multiple frames passed to input of SAC
            #  pytorch
            log.info(f'm/s {round(self.speed, 2)} '
                     f'deviation {round(lane_deviation, 2)}')
            if lane_deviation < 1 and self.speed > 2:  # ~5mph
                # TODO: Double check these are meters
                if CONTINUOUS_REWARD:
                    max_desired_speed = 4  # ~20mph
                    min_desired_speed = 2  # ~5mph
                    reward = (self.speed - min_desired_speed) / (
                                max_desired_speed - min_desired_speed)
                    reward = min(reward, 1)
                    log.info(f'Reward: {reward}')
                else:
                    reward = 1
            else:
                # Avoid negative rewards due to
                # http://bit.ly/mistake_importance_scaling
                reward = 0

            if lane_deviation > 1.1:
                # You lose!
                done = True

            if list(self.map.arr[-1]) == list(closest_map_point):
                # You win!
                done = True

        self.last_step_time = time.time()
        return observation, reward, done, info

    def get_observation(self, accel, brake, dt, start, steer):
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
        closest_map_point, lane_deviation = \
            get_closest_point((self.x, self.y), self.map_kd_tree)
        log.trace(f'time {time.time() - start}')
        observation = self.populate_observation(closest_map_point,
                                                lane_deviation)
        return lane_deviation, observation, closest_map_point

    def populate_observation(self, closest_map_point, lane_deviation):
        observation = \
            Box(x=self.x,
                y=self.y,
                angle=self.angle,
                closest_map_point_x=closest_map_point[0],
                closest_map_point_y=closest_map_point[1],
                lane_deviation=lane_deviation,
                vehicle_width=self.vehicle_width,
                vehicle_height=self.vehicle_height,
                cog_to_front_axle=self.vehicle_model[0],
                cog_to_rear_axle=self.vehicle_model[1], )
        if self.return_observation_as_array:
            observation = observation.values()
            observation += self.map_flat
        return observation

    def reset(self):
        self.step_num = 0
        ret = self.get_blank_observation()
        return ret

    def get_blank_observation(self):
        ret = self.populate_observation(closest_map_point=self.map.arr[0],
                                        lane_deviation=0)
        return ret

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
    return point, distance
