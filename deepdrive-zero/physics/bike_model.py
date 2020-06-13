from math import pi, cos, sin

import numpy as np
from numba import njit

from deepdrive_zero.constants import USE_VOYAGE, VEHICLE_WIDTH, CACHE_NUMBA

TUNED_FPS = 1 / 60  # The FPS we tuned friction ratios at
ROTATIONAL_FRICTION = 0.95
LONGITUDINAL_FRICTION = 0.999
BRAKE_FRICTION = 0.96


@njit(cache=CACHE_NUMBA, nogil=True)
def bike_with_friction_step(
        steer,
        accel,
        brake,
        dt,
        x,
        y,
        angle,
        angle_change,
        velocity,
        add_rotational_friction,
        add_longitudinal_friction,
        vehicle_model):
    """

    :param steer: (float) Steering angle in radians
    :param accel: (float) m/s**2
    :param brake: (float) whether to brake or not
    :param dt: (float) time step
    :param x: (float) meters x
    :param y: (float) meters y
    :param angle: (float) radian angle offset
    :param angle_change: (float) angular velocity
    :param velocity: (float) m/s
    :param add_rotational_friction: (bool)
        Whether to slow angular velocity every time step
    :param add_longitudinal_friction: (bool)
        Whether to slow velocity every time step
    :param vehicle_model: (List)
        Distance from center of gravity to front and rear axles
    :return:
    """
    steer = min(pi, steer)
    steer = max(-pi, steer)

    change_x, change_y, angle_change, velocity = \
        f_KinBkMdl(angle_change, velocity, steer, accel, vehicle_model, dt)

    friction_exponent = (dt / TUNED_FPS)
    if add_rotational_friction:
        # Causes steering to drift back to zero
        angle_change *= ROTATIONAL_FRICTION ** friction_exponent
    if add_longitudinal_friction:
        velocity *= LONGITUDINAL_FRICTION ** friction_exponent

    velocity = BRAKE_FRICTION ** (brake * friction_exponent) * velocity

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

    return x, y, angle, angle_change, velocity


@njit(cache=CACHE_NUMBA, nogil=True)
def f_KinBkMdl(velocity, angle_velocity, steer_angle, accel, vehicle_model, dt):
    """
    Computes change in x,y and next velocity, angular velocity for a
    given control
    :param velocity: (float) Velocity of car along its heading
    :param angle_velocity: (float) Rotational velocity in radians per second
        about the center of gravity
    :param steer_angle: (float) Steering angle in radians with respect to world
    :param accel: (float) Desired forward acceleration in m/s^2
    :param vehicle_model: (List)
        Distance from center of gravity to front and rear axles
    # From https://github.com/MPC-Berkeley/barc/blob/master/workspace/src/barc/src/estimation/system_models.py
    c.f.: https://www.coursera.org/lecture/intro-self-driving-cars/lesson-2-the-kinematic-bicycle-model-Bi8yE
    """
    # extract parameters
    # Distance from center of gravity to front and rear axles
    (cg_to_front_axle, cg_to_rear_axle) = vehicle_model

    # compute slip angle
    slip = cg_to_front_axle / (cg_to_front_axle + cg_to_rear_axle)
    slip_angle = np.arctan(slip * np.tan(steer_angle))

    # compute next state
    change_x = dt * (velocity * np.cos(angle_velocity + slip_angle))
    change_y = dt * (velocity * np.sin(angle_velocity + slip_angle))
    angle_velocity += dt * velocity / cg_to_rear_axle * np.sin(slip_angle)
    velocity += dt * accel

    return np.array([change_x, change_y, angle_velocity, velocity])


def get_vehicle_model(length):
    """
    :param length: Length of vehicle
    :return: Distance from center of gravity to front and rear axles
    """
    # Bias towards the front a bit
    # https://www.fcausfleet.com/content/dam/fca-fleet/na/fleet/en_us/chrysler/2017/pacifica/vlp/docs/Pacifica_Specifications.pdf

    if USE_VOYAGE and False:
        # N.B. Disabled until rendering rotates around this point, or bike model
        # is changed to return angle around geometric center vs center of
        # gravity
        bias_towards_front = .05 * length
    else:
        bias_towards_front = 0

    # Center of gravity
    center_of_gravity = (length / 2) + bias_towards_front
    # Approximate axles to be 1/8 (1/4 - 1/8) from ends of car
    rear_axle = length * 1 / 8
    front_axle = length - rear_axle
    L_b = center_of_gravity - rear_axle
    L_a = front_axle - center_of_gravity
    return L_a, L_b


def test_bike_with_friction_step():
    vehicle_model = get_vehicle_model(VEHICLE_WIDTH)

    # Do nothing
    x, y, angle, angle_change, speed = bike_with_friction_step(
        steer=0, accel=0, brake=0, dt=1, x=0, y=0, angle=0, angle_change=0,
        velocity=0, add_longitudinal_friction=True, add_rotational_friction=True,
        vehicle_model=vehicle_model,)

    assert x == 0
    assert y == 0
    assert angle == 0
    assert speed == 0
    assert angle_change == 0

    # Just steer, should be no change still
    x, y, angle, angle_change, speed = bike_with_friction_step(
        steer=pi/4, accel=0, brake=0, dt=1, x=0, y=0, angle=0, angle_change=0,
        velocity=0, add_longitudinal_friction=True, add_rotational_friction=True,
        vehicle_model=vehicle_model,)

    assert x == 0
    assert y == 0
    assert angle == 0
    assert speed == 0
    assert angle_change == 0

    # Just steer, should be no change still
    x, y, angle, angle_change, speed = bike_with_friction_step(
        steer=0, accel=0, brake=0, dt=1, x=0, y=0, angle=0, angle_change=0,
        velocity=1, add_longitudinal_friction=True, add_rotational_friction=True,
        vehicle_model=vehicle_model,)

    assert np.isclose(x, 0) and x == 6.123233995736766e-17
    assert y == 1
    assert angle == 0
    assert speed == 0.9417362622231682
    assert angle_change == 0


if __name__ == '__main__':
    test_bike_with_friction_step()
