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
        speed,
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
    :param speed: (float) m/s
    :param add_rotational_friction: (bool)
        Whether to slow angular velocity every time step
    :param add_longitudinal_friction: (bool)
        Whether to slow velocity every time step
    :param vehicle_model:
        Distance from center of gravity to front and rear axles
    :return:
    """
    steer = min(pi, steer)
    steer = max(-pi, steer)
    state = [x, y, angle_change, speed]

    change_x, change_y, angle_change, speed = \
        f_KinBkMdl(state, steer, accel, vehicle_model, dt)

    friction_exponent = (dt / TUNED_FPS)
    if add_rotational_friction:
        # Causes steering to drift back to zero
        angle_change *= ROTATIONAL_FRICTION ** friction_exponent
    if add_longitudinal_friction:
        speed *= LONGITUDINAL_FRICTION ** friction_exponent

    # TODO: We should just output the desired accel (+-) and allow higher magnitude
    #   negative accel than positive due to brake force. Otherwise network will
    #   be riding brakes more than reasonable (esp without disincentivizing this
    #   in the reward)
    speed = BRAKE_FRICTION ** (brake * friction_exponent) * speed

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


@njit(cache=CACHE_NUMBA, nogil=True)
def f_KinBkMdl(state, steer_angle, accel, vehicle_model, dt):
    """
    process model
    input: state at time k, z[k] := [x[k], y[k], angle_change[k], v[k]]
    output: state of center of gravity at next time step z[k+1]
    # From https://github.com/MPC-Berkeley/barc/blob/master/workspace/src/barc/src/estimation/system_models.py
    c.f.: https://www.coursera.org/lecture/intro-self-driving-cars/lesson-2-the-kinematic-bicycle-model-Bi8yE
    """

    # get states / inputs
    x = state[0]             # straight
    y = state[1]             # right
    angle_speed = state[2]  # radians per second
    speed = state[3]         # meters per second

    # extract parameters
    # Distance from center of gravity to front and rear axles
    (cg_to_front_axle, cg_to_rear_axle) = vehicle_model

    # compute slip angle
    slip = cg_to_front_axle / (cg_to_front_axle + cg_to_rear_axle)
    slip_angle = np.arctan(slip * np.tan(steer_angle))

    # compute next state
    change_x = dt * (speed * np.cos(angle_speed + slip_angle))
    change_y = dt * (speed * np.sin(angle_speed + slip_angle))
    angle_speed += dt * speed / cg_to_rear_axle * np.sin(slip_angle)
    speed += dt * accel

    return np.array([change_x, change_y, angle_speed, speed])


# @njit(cache=CACHE_NUMBA, nogil=True)
def get_angle_for_accel(desired_accel, speed, vehicle_model, prev_angle_change,
                        aps, angle, prev_accel):

    """
    Returns change in angle that will result in the desired rotational
    acceleration due to change in velocity vector direction for next step,
    in other words the inverse bike model for steering.

    Based on velocity magnitude (s) change per change in
    angle (a)
    i.e. ds = ||[cos(a)*s - s, sin(a) * s]||2
    remember: cos^2 + sin^2 = 1

    NOT WORKING
    """

    if abs(speed) < 0.075:
        # Need speed to turn!
        return 0

    dt = 1/aps

    # # Get previous slip angle
    # cg_to_front_axle, cg_to_rear_axle = vehicle_model  # cg = Center of gravity
    # slip = cg_to_front_axle / (cg_to_front_axle + cg_to_rear_axle)
    # slip_angle = np.arctan(slip * np.tan(angle))

    # # Subtract current rotational acceleration
    # desired_accel -= dt * speed / cg_to_rear_axle * np.sin(slip_angle)

    # # Subtract current linear acceleration
    # desired_accel -= prev_accel

    # friction_exponent = (dt / TUNED_FPS)
    # rot_friction_accel = ROTATIONAL_FRICTION ** friction_exponent * prev_angle_change
    #
    # long_friction_accel = LONGITUDINAL_FRICTION ** friction_exponent * speed
    #
    # # Account for accel lost due to friction
    # additional_accel += rot_friction_accel + long_friction_accel

    # Get angle that will lead to desired rotational accel
    desired_accel /= ((speed + 1) ** 0.75 * 0.4)   # Magic tuning
    cos_theta = 1 - (dt * desired_accel) ** 2 / (2 * speed ** 2)
    theta = np.arccos(np.clip(cos_theta, -1, 1))

    # theta /= speed

    # prev_angle_change /= dt  # magic, works for 1g only and not stable!

    # theta /= 2

    return theta


def get_angle_for_accel_track(target_accel, current_accel, prev_steer, dt, speed):
    # Slowly add more steer until we get to desired
    delta = target_accel - current_accel
    ret = 0
    if delta > 0:
        ret = min(prev_steer + 1 / (1 + prev_steer*2e3), prev_steer + 0.005)
    elif delta < 0:
        ret = prev_steer * 0.9
    return ret


def get_angle_for_accel_pid(target_accel, current_accel, current_angle,
                            current_angle_change):
    if pid.setpoint != target_accel:
        pid.setpoint = target_accel
    desired_angle = pid(current_accel)
    if not pid.auto_mode:
        pid.auto_mode = True
    if throttle is None:
        log.warn('PID output None, setting throttle to 0.')
        throttle = 0.
    throttle = min(max(throttle, 0.), 1.)
    return throttle


def get_vehicle_model(width):
    """
    :param width: Width of vehicle  # TODO: Should be length! Change after ablation test
    :return: Distance from center of gravity to front and rear axles
    """
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


def test_bike_with_friction_step():
    vehicle_model = get_vehicle_model(VEHICLE_WIDTH)

    # Do nothing
    x, y, angle, angle_change, speed = bike_with_friction_step(
        steer=0, accel=0, brake=0, dt=1, x=0, y=0, angle=0, angle_change=0,
        speed=0, add_longitudinal_friction=True, add_rotational_friction=True,
        vehicle_model=vehicle_model,)

    assert x == 0
    assert y == 0
    assert angle == 0
    assert speed == 0
    assert angle_change == 0

    # Just steer, should be no change still
    x, y, angle, angle_change, speed = bike_with_friction_step(
        steer=pi/4, accel=0, brake=0, dt=1, x=0, y=0, angle=0, angle_change=0,
        speed=0, add_longitudinal_friction=True, add_rotational_friction=True,
        vehicle_model=vehicle_model,)

    assert x == 0
    assert y == 0
    assert angle == 0
    assert speed == 0
    assert angle_change == 0

    # Just steer, should be no change still
    x, y, angle, angle_change, speed = bike_with_friction_step(
        steer=0, accel=0, brake=0, dt=1, x=0, y=0, angle=0, angle_change=0,
        speed=1, add_longitudinal_friction=True, add_rotational_friction=True,
        vehicle_model=vehicle_model,)

    assert np.isclose(x, 0) and x == 6.123233995736766e-17
    assert y == 1
    assert angle == 0
    assert speed == 0.9417362622231682
    assert angle_change == 0


if __name__ == '__main__':
    test_bike_with_friction_step()
