import math

import numpy as np
from numba import njit

from deepdrive_zero.constants import CACHE_NUMBA, MAX_STEER_CHANGE_PER_SECOND, \
    MAX_ACCEL_CHANGE_PER_SECOND
from deepdrive_zero.physics.bike_model import bike_with_friction_step

@njit(cache=CACHE_NUMBA, nogil=True)
def physics_step(throttle,
                 add_longitudinal_friction,
                 add_rotational_friction,
                 brake,
                 constrain_controls,
                 acceleration,
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
                 max_accel_change,
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
        accel = 0
    steer_change = steer - prev_steer
    accel_change = accel - prev_accel
    if constrain_controls:
        steer_change = min(max_steer_change, steer_change)
        steer_change = max(-max_steer_change, steer_change)
        accel_change = min(max_accel_change, accel_change)
        accel_change = max(-max_accel_change, accel_change)

        i_steer, i_throttle, i_brake = 0, 0, 0

    start_x = curr_x
    start_y = curr_y
    for i in range(interpolation_steps):
        """
        Enforce real-world constraint that you can't teleport the gas pedal
        or steering wheel between positions, rather you must visit the
        intermediate positions between subsequent settings.
        """
        interp = (start_interpolation_index + i + 1) / interpolation_range
        i_steer = prev_steer + interp * steer_change
        i_accel = prev_accel + interp * accel_change
        if brake:
            i_brake = prev_brake + interp * (brake - prev_brake)
        else:
            i_brake = 0
        # log.info(f'steer {steer} accel {accel} brake {brake} vel {self.velocity}')
        # TODO: Add drag when simulating higher speeds
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
         acceleration,
         new_jerk,
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
            acceleration,
            jerk,)

    return (acceleration,
            curr_angle,
            curr_angle_change,
            curr_angular_velocity,
            curr_gforce,
            new_jerk,
            curr_max_gforce,
            curr_max_jerk,
            curr_speed,
            curr_x,
            curr_y,
            i_throttle,
            i_brake,
            i_steer,
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
    prev_velocity = curr_velocity
    curr_velocity = pos_change / dt
    new_accel = (curr_velocity - prev_velocity) / dt
    angular_velocity = (curr_angle - prev_angle) / dt
    accel_magnitude = np.linalg.norm(new_accel)
    gforce = accel_magnitude / 9.807
    max_gforce = max(gforce, curr_max_gforce)
    # prev_gforce.append(self.gforce)
    jerk = (new_accel - curr_accel) / dt
    jerk_magnitude = np.linalg.norm(jerk)
    max_jerk = max(jerk_magnitude, curr_max_jerk)
    return (gforce, max_gforce, max_jerk, jerk, new_accel, angular_velocity,
            curr_velocity)