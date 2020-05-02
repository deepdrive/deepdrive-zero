import math
from copy import deepcopy

import numpy as np
from box import Box
from numba import njit

from deepdrive_zero.constants import CACHE_NUMBA, MAX_STEER_CHANGE_PER_SECOND, \
    MAX_ACCEL_CHANGE_PER_SECOND, VEHICLE_WIDTH
from deepdrive_zero.physics.bike_model import bike_with_friction_step

@njit(cache=CACHE_NUMBA, nogil=True)
def physics_step(throttle,
                 add_longitudinal_friction,
                 add_rotational_friction,
                 brake,
                 constrain_controls,
                 curr_acceleration,
                 jerk,
                 curr_angle,
                 curr_angle_change,
                 curr_angular_velocity,
                 curr_gforce,
                 curr_max_gforce,
                 curr_max_jerk,
                 curr_speed,
                 curr_velocity,
                 curr_x,
                 curr_y,
                 distance_traveled,
                 dt,
                 ignore_brake,
                 max_throttle_change,
                 max_brake_change,
                 max_steer_change,
                 interpolation_steps,
                 prev_throttle,
                 prev_brake,
                 prev_steer,
                 steer,
                 vehicle_model,
                 start_interpolation_index,
                 interpolation_range, ):
    if ignore_brake:
        brake = 0
    if curr_speed > 100:
        throttle = 0
    steer_change = steer - prev_steer
    throttle_change = throttle - prev_throttle
    brake_change = brake - prev_brake
    if constrain_controls:
        steer_change = min(max_steer_change, steer_change)
        steer_change = max(-max_steer_change, steer_change)
        throttle_change = min(max_throttle_change, throttle_change)
        throttle_change = max(-max_throttle_change, throttle_change)

        i_steer, i_throttle, i_brake = 0, 0, 0

    start_x = curr_x
    start_y = curr_y
    out_steer = prev_steer
    out_throttle = prev_throttle
    out_brake = prev_brake
    for i in range(interpolation_steps):
        """
        Enforce real-world constraint that you can't teleport the gas pedal
        or steering wheel between positions, rather you must visit the
        intermediate positions between subsequent settings.
        """
        interp_index = start_interpolation_index + i
        interp = (interp_index + 1) / interpolation_range
        i_steer = prev_steer + interp * steer_change
        i_throttle = prev_throttle + interp * throttle_change
        if brake:
            i_brake = prev_brake + interp * brake_change
        else:
            i_brake = 0
        # log.info(f'steer {steer} accel {accel} brake {brake} vel {self.velocity}')

        prev_x, prev_y, prev_angle = curr_x, curr_y, curr_angle

        (curr_angle,
         curr_angle_change,
         curr_angular_velocity,
         curr_gforce,
         curr_max_gforce,
         curr_max_jerk,
         curr_speed,
         curr_x,
         curr_y,
         distance_traveled,
         curr_acceleration,
         curr_jerk,
         curr_velocity) = interp_physics_step(
            throttle,
            add_longitudinal_friction,
            add_rotational_friction,
            curr_angle,
            curr_angle_change,
            curr_angular_velocity,
            curr_gforce,
            curr_max_gforce,
            curr_max_jerk,
            curr_speed,
            curr_velocity,
            curr_x,
            curr_y,
            distance_traveled,
            dt,
            i_throttle,
            i_brake,
            i_steer,
            prev_angle,
            prev_x,
            prev_y,
            vehicle_model,
            curr_acceleration,
            jerk,)

        if interp_index == interpolation_range - 1:
            out_steer = i_steer
            out_throttle = i_throttle
            out_brake = i_brake

    return (curr_acceleration,
            curr_angle,
            curr_angle_change,
            curr_angular_velocity,
            curr_gforce,
            curr_jerk,
            curr_max_gforce,
            curr_max_jerk,
            curr_speed,
            curr_x,
            curr_y,
            out_throttle,
            out_brake,
            out_steer,
            curr_velocity,
            distance_traveled)


@njit(cache=CACHE_NUMBA, nogil=True)
def interp_physics_step(throttle,
                        add_longitudinal_friction,
                        add_rotational_friction,
                        angle,
                        angle_change,
                        angular_velocity,
                        gforce,
                        max_gforce,
                        max_jerk,
                        speed,
                        velocity,
                        x,
                        y,
                        distance_traveled,
                        dt,
                        i_accel,
                        i_brake,
                        i_steer,
                        prev_angle,
                        prev_x,
                        prev_y,
                        vehicle_model,
                        acceleration,
                        jerk,):
    (x,
     y,
     angle,
     angle_change,
     speed) = bike_with_friction_step(
        steer=i_steer,
        accel=i_accel,
        brake=i_brake,
        dt=dt,
        x=x,
        y=y,
        angle=angle,
        angle_change=angle_change,
        speed=speed,
        add_rotational_friction=add_rotational_friction,
        add_longitudinal_friction=add_longitudinal_friction,
        vehicle_model=vehicle_model,)

    distance_traveled += np.linalg.norm(
        np.array([x - prev_x, y - prev_y]))

    (gforce,
     max_gforce,
     max_jerk,
     jerk,
     acceleration,
     angular_velocity,
     velocity) = get_gforce_levels(x=x,
                                   y=y,
                                   angle=angle,
                                   prev_x=prev_x,
                                   prev_y=prev_y,
                                   prev_angle=prev_angle,
                                   dt=dt,
                                   prev_velocity=velocity,
                                   prev_acceleration=acceleration,
                                   max_gforce=max_gforce,
                                   max_jerk=max_jerk)
    return (angle,
            angle_change,
            angular_velocity,
            gforce,
            max_gforce,
            max_jerk,
            speed,
            x,
            y,
            distance_traveled,
            acceleration,
            jerk,
            velocity)


@njit(cache=CACHE_NUMBA, nogil=True)
def get_gforce_levels(x,
                      y,
                      angle,
                      prev_x,
                      prev_y,
                      prev_angle,
                      dt,
                      prev_velocity,
                      prev_acceleration,
                      max_gforce,
                      max_jerk):
    pos_change = np.array([prev_x - x, prev_y - y])
    velocity = pos_change / dt
    acceleration = (velocity - prev_velocity) / dt
    angular_velocity = (angle - prev_angle) / dt
    accel_magnitude = np.linalg.norm(acceleration)
    gforce = accel_magnitude / 9.807
    max_gforce = max(gforce, max_gforce)
    jerk = (acceleration - prev_acceleration) / dt
    jerk_magnitude = np.linalg.norm(jerk)
    max_jerk = max(jerk_magnitude, max_jerk)
    return (gforce, max_gforce, max_jerk, jerk, acceleration, angular_velocity,
            velocity)


def test_physics_step():
    from deepdrive_zero.physics.bike_model import get_vehicle_model
    vehicle_model = get_vehicle_model(VEHICLE_WIDTH)
    zero2d = np.array((0,0), dtype=np.float64)

    # Do 1 step with 12 pso
    # Do 12 steps with 1 interp
    physics_steps_per_observation = 12
    state_interp = Box(
        throttle=1.,
        add_longitudinal_friction=True,
        add_rotational_friction=True,
        brake=0.,
        curr_acceleration=zero2d,
        jerk=zero2d,
        curr_angle=0.,
        curr_angle_change=0.,
        curr_angular_velocity=0.,
        curr_gforce=0.,
        curr_max_gforce=0.,
        curr_max_jerk=0.,
        curr_speed=0.,
        curr_velocity=zero2d,
        curr_x=0.,
        curr_y=0.,
        dt=1 / 60,
        interpolation_steps=1,
        prev_throttle=0,
        prev_brake=0,
        prev_steer=0,
        steer=1.,
        vehicle_model=vehicle_model,
        ignore_brake=False,
        constrain_controls=False,
        max_steer_change=0.,
        max_throttle_change=0.,
        max_brake_change=0.,
        distance_traveled=0.,
        start_interpolation_index=0,
        interpolation_range=physics_steps_per_observation,
    )
    state_one_shot = deepcopy(state_interp)
    state_one_shot.interpolation_steps = 12
    for _ in range(10):
        for _ in range(physics_steps_per_observation):
            interpolation_steps = 1
            state_interp = run_test_step(state_interp)
            state_interp.start_interpolation_index += 1
        state_one_shot = run_test_step(state_one_shot)
    known_differences = {'interpolation_steps', 'start_interpolation_index'}
    for k, v in state_interp.items():
        if k not in known_differences and not np.allclose(v, state_one_shot[k]):
            raise RuntimeError(f'Interp and One shot states not equal for '
                               f'"{k}" interp={v}, one_shot={state_one_shot[k]}')



def run_test_step(state):
    (curr_acceleration,
     curr_angle,
     curr_angle_change,
     curr_angular_velocity,
     curr_gforce,
     curr_jerk,
     curr_max_gforce,
     curr_max_jerk,
     curr_speed,
     curr_x,
     curr_y,
     out_throttle,
     out_brake,
     out_steer,
     curr_velocity,
     distance_traveled) = physics_step(**state)

    state.curr_acceleration = curr_acceleration
    state.curr_angle = curr_angle
    state.curr_angle_change = curr_angle_change
    state.curr_angular_velocity = curr_angular_velocity
    state.curr_gforce = curr_gforce
    state.jerk = curr_jerk
    state.curr_max_gforce = curr_max_gforce
    state.curr_max_jerk = curr_max_jerk
    state.curr_speed = curr_speed
    state.curr_x = curr_x
    state.curr_y = curr_y
    state.prev_throttle = out_throttle
    state.prev_brake = out_brake
    state.prev_steer = out_steer
    state.curr_velocity = curr_velocity
    state.distance_traveled = distance_traveled

    return state


if __name__ == '__main__':
    test_physics_step()
