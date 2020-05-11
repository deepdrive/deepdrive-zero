import math
from collections import deque
from math import cos, sin, pi
import os
import sys
from random import random
from typing import List

import numpy as np

from loguru import logger as log

import arcade
import arcade.color as color
from deepdrive_zero.constants import SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_MARGIN, \
    MAP_WIDTH_PX, MAP_HEIGHT_PX, PLAYER_TURN_RADIANS_PER_KEYSTROKE, \
    SCREEN_TITLE, \
    CHARACTER_SCALING, MAX_PIXELS_PER_SEC_SQ, TESLA_LENGTH, VOYAGE_VAN_LENGTH, \
    USE_VOYAGE, VEHICLE_PNG, MAX_METERS_PER_SEC_SQ, MAP_IMAGE, \
    PARTIAL_PHYSICS_STEP
# Constants
from deepdrive_zero.envs.env import Deepdrive2DEnv
from deepdrive_zero.map_gen import get_intersection

DRAW_COLLISION_BOXES = True
DRAW_WAYPOINT_VECTORS = False
DRAW_INTERSECTION = True

# TODO: Calculate rectangle points and confirm corners are at same location in
#   arcade.


# noinspection PyAbstractClass
class Deepdrive2DPlayer(arcade.Window):
    """Allows playing the env as a human"""
    def __init__(self, add_rotational_friction=True,
                 add_longitudinal_friction=True, env=None,
                 fps=60, static_obstacle=False, one_waypoint=False,
                 is_intersection_map=False, env_config=None,
                 end_on_lane_violation=False):

        # Call the parent class and set up the window
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE,
                         update_rate=1/fps)
        self.add_rotational_friction = add_rotational_friction
        self.add_longitudinal_friction = add_longitudinal_friction
        self.fps = fps

        arcade.set_background_color(arcade.csscolor.CORNFLOWER_BLUE)
        self.player_list = None
        self.human_controlled = False if env else True
        self.env: Deepdrive2DEnv = env
        self.steer = 0
        self.accel = 0
        self.brake = 0
        self.map = None
        self.angle = None
        self.background = None
        self.max_accel = None
        self.px_per_m = None
        self.static_obstacle = static_obstacle
        self.is_intersection_map = is_intersection_map
        self.one_waypoint = one_waypoint
        self.end_on_lane_violation = end_on_lane_violation

        self.env_config = env_config
        self.updates = 0
        self.physics_generators = None
        self.prev_step_out = None

    def setup(self):
        """ Set up the game here. Call this function to restart the game. """
        self.player_list = arcade.SpriteList()
        # self.background = arcade.load_texture(MAP_IMAGE)

        vehicle_length_pixels = arcade.Sprite(
            VEHICLE_PNG, CHARACTER_SCALING).height
        if USE_VOYAGE:
            vehicle_length_meters = VOYAGE_VAN_LENGTH
        else:
            vehicle_length_meters = TESLA_LENGTH
        self.px_per_m = vehicle_length_pixels / vehicle_length_meters
        self.max_accel = MAX_PIXELS_PER_SEC_SQ / self.px_per_m

        if self.env is None:
            self.env = Deepdrive2DEnv(
                px_per_m=self.px_per_m,
                add_rotational_friction=self.add_rotational_friction,
                add_longitudinal_friction=self.add_longitudinal_friction,
                return_observation_as_array=False,
                expect_normalized_actions=False,
                expect_normalized_action_deltas=False,
                decouple_step_time=True,
                physics_steps_per_observation=1,
                add_static_obstacle=self.static_obstacle,
                is_one_waypoint_map=self.one_waypoint,
                is_intersection_map=self.is_intersection_map,
                being_played=True,
            )

        self.env.configure_env(self.env_config)

        self.static_obstacle = (self.static_obstacle or
                                (self.env and
                                 self.env.unwrapped.add_static_obstacle))
        self.is_intersection_map = (self.is_intersection_map or
                                    (self.env and
                                     self.env.unwrapped.is_intersection_map))
        self.one_waypoint = (self.one_waypoint or
                             (self.env and
                              self.env.unwrapped.is_one_waypoint_map))

        self.env.reset()

        self.physics_generators = []
        agents = self.env.all_agents
        for i, agent in enumerate(agents):
            sprite = arcade.Sprite(VEHICLE_PNG, CHARACTER_SCALING)
            sprite.center_x = agent.map.x_pixels[0]
            sprite.center_y = agent.map.y_pixels[0]
            self.player_list.append(sprite)
            self.physics_generators.append(None)

    def on_draw(self):
        arcade.start_render()

        for agent in self.env.all_agents:
            self.draw_agent_objects(agent)

        if self.is_intersection_map:
            self.draw_intersection()

        # arcade.draw_line(300, 300, 300 + self.player_sprite.height, 300,
        #                  arcade.color.WHITE)
        # arcade.draw_lines(self.map, arcade.color.ORANGE, 3)
        # arcade.draw_point(self.heading_x, self.heading_y,
        #                   arcade.color.WHITE, 10)

        self.player_list.draw()  # Draw the car

    def draw_agent_objects(self, agent):
        a = agent
        e = self.env
        m = a.map
        ppm = e.px_per_m
        angle = a.angle
        theta = angle + pi / 2
        if self.env.is_one_waypoint_map:
            arcade.draw_circle_filled(
                center_x=m.x_pixels[1],
                center_y=m.y_pixels[1],
                radius=21,
                color=color.ORANGE)
            if self.static_obstacle:
                static_obst_pixels = m.static_obst_pixels
                arcade.draw_line(
                    static_obst_pixels[0][0],
                    static_obst_pixels[0][1],
                    static_obst_pixels[1][0],
                    static_obst_pixels[1][1],
                    color=color.BLACK_OLIVE,
                    line_width=5,
                )
        elif self.is_intersection_map:
            if agent.agent_index == 0:
                wp_clr = (10, 210, 50)
            else:
                wp_clr = (250, 140, 20)
            for i in range(len(m.waypoints)):
                arcade.draw_circle_filled(
                    center_x=m.x_pixels[i],
                    center_y=m.y_pixels[i],
                    radius=21,
                    color=wp_clr)
        else:
            # Draw the background texture
            bg_scale = 1.1
            arcade.draw_texture_rectangle(
                MAP_WIDTH_PX // 2 + SCREEN_MARGIN,
                MAP_HEIGHT_PX // 2 + SCREEN_MARGIN,
                MAP_WIDTH_PX * bg_scale,
                MAP_HEIGHT_PX * bg_scale,
                self.background)
        if a.ego_rect is not None and DRAW_COLLISION_BOXES:
            arcade.draw_rectangle_outline(
                center_x=a.x * ppm, center_y=a.y * ppm,
                width=a.vehicle_width * ppm,
                height=a.vehicle_length * ppm, color=color.LIME_GREEN,
                border_width=2, tilt_angle=math.degrees(a.angle),
            )
            arcade.draw_points(point_list=(a.ego_rect * ppm).tolist(),
                               color=color.YELLOW, size=3)
        if a.front_to_waypoint is not None and DRAW_WAYPOINT_VECTORS:
            ftw = a.front_to_waypoint

            fy = a.front_y
            fx = a.front_x

            # arcade.draw_line(
            #     start_x=e.front_x * ppm,
            #     start_y=e.front_y * ppm,
            #     end_x=(e.front_x + ftw[0]) * ppm,
            #     end_y=(e.front_y + ftw[1]) * ppm,
            #     color=c.LIME_GREEN,
            #     line_width=2,
            # )

            arcade.draw_line(
                start_x=fx * ppm,
                start_y=fy * ppm,
                end_x=(fx + cos(
                    theta - a.angle_to_waypoint) * a.distance_to_end) * ppm,
                end_y=(fy + sin(
                    theta - a.angle_to_waypoint) * a.distance_to_end) * ppm,
                color=color.PURPLE,
                line_width=2,
            )

            # Center to front length
            # ctf = a.vehicle_length / 2

            arcade.draw_line(
                start_x=a.x * ppm,
                start_y=a.y * ppm,
                end_x=(a.x + cos(theta) * 20) * ppm,
                end_y=(a.y + sin(theta) * 20) * ppm,
                color=color.LIGHT_RED_OCHRE,
                line_width=2,
            )

            arcade.draw_line(
                start_x=fx * ppm,
                start_y=fy * ppm,
                end_x=(fx + a.heading[0]) * ppm,
                end_y=(fy + a.heading[1]) * ppm,
                color=color.BLUE,
                line_width=2,
            )

            arcade.draw_circle_filled(
                center_x=fx * ppm,
                center_y=fy * ppm,
                radius=5,
                color=color.YELLOW)

            arcade.draw_circle_filled(
                center_x=a.x * ppm,
                center_y=a.y * ppm,
                radius=5,
                color=color.WHITE_SMOKE, )

            arcade.draw_circle_filled(
                center_x=a.static_obstacle_points[0][0] * ppm,
                center_y=a.static_obstacle_points[0][1] * ppm,
                radius=5,
                color=color.WHITE_SMOKE, )

            arcade.draw_circle_filled(
                center_x=a.static_obstacle_points[1][0] * ppm,
                center_y=a.static_obstacle_points[1][1] * ppm,
                radius=5,
                color=color.WHITE_SMOKE, )

            if a.static_obst_angle_info is not None:
                start_obst_dist, end_obst_dist, start_obst_angle, end_obst_angle = \
                    a.static_obst_angle_info

                # start_obst_theta = start_obst_angle
                # arcade.draw_line(
                #     start_x=fx * ppm,
                #     start_y=fy * ppm,
                #     end_x=(fx + cos(start_obst_theta) * start_obst_dist) * ppm,
                #     end_y=(fy + sin(start_obst_theta) * start_obst_dist) * ppm,
                #     color=c.BLACK,
                #     line_width=2,)

                # log.info('DRAWING LINES')

                arcade.draw_line(
                    start_x=fx * ppm,
                    start_y=fy * ppm,
                    end_x=(fx + cos(
                        theta - start_obst_angle) * start_obst_dist) * ppm,
                    end_y=(fy + sin(
                        theta - start_obst_angle) * start_obst_dist) * ppm,
                    color=color.BLUE,
                    line_width=2, )

                p_x = a.front_x + cos(theta + pi / 6) * 20
                p_y = a.front_y + sin(theta + pi / 6) * 20
                pole_test = np.array((p_x, p_y))
                pole_angle = a.get_angle_to_point(pole_test)

                arcade.draw_circle_filled(
                    center_x=pole_test[0] * ppm,
                    center_y=pole_test[1] * ppm,
                    radius=5,
                    color=color.WHITE_SMOKE, )

                arcade.draw_line(
                    start_x=fx * ppm,
                    start_y=fy * ppm,
                    end_x=(fx + cos(
                        (angle + math.pi / 2) - pole_angle) * 20) * ppm,
                    end_y=(fy + sin(
                        (angle + math.pi / 2) - pole_angle) * 20) * ppm,
                    color=color.BRIGHT_GREEN,
                    line_width=2, )

                # arcade.draw_line(
                #     start_x=fx * ppm,
                #     start_y=fy * ppm,
                #     end_x=(fx + cos((angle + math.pi / 2) - end_obst_angle) * end_obst_dist) * ppm,
                #     end_y=(fy + sin((angle + math.pi / 2) - end_obst_angle) * end_obst_dist) * ppm,
                #     color=c.RED,
                #     line_width=2,)

                arcade.draw_line(
                    start_x=fx * ppm,
                    start_y=fy * ppm,
                    end_x=(a.static_obstacle_points[1][0]) * ppm,
                    end_y=(a.static_obstacle_points[1][1]) * ppm,
                    color=color.RED,
                    line_width=2, )

    def draw_intersection(self):
        lines, lane_width = get_intersection()

        bottom_horiz, left_vert, mid_horiz, mid_vert, right_vert, top_horiz = lines
        self.draw_intersection_line(left_vert)
        self.draw_intersection_line(mid_vert)
        self.draw_intersection_line(right_vert)
        self.draw_intersection_line(top_horiz)
        self.draw_intersection_line(mid_horiz)
        self.draw_intersection_line(bottom_horiz)

    def draw_intersection_line(self, line):
        line = line * self.px_per_m
        arcade.draw_line(
            line[0][0], line[0][1], line[1][0], line[1][1],
            color=(100, 200, 240),
            line_width=2,
        )

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """

        print('key pressed!')
        if key == arcade.key.UP or key == arcade.key.W:
            self.accel = MAX_METERS_PER_SEC_SQ
        elif key == arcade.key.DOWN or key == arcade.key.S:
            self.accel = -MAX_METERS_PER_SEC_SQ
        elif key == arcade.key.SPACE:
            self.brake = 1

        # Don't divide by physics steps here as it
        # restricts our max turning radius
        elif key == arcade.key.LEFT or key == arcade.key.A:
            self.steer = PLAYER_TURN_RADIANS_PER_KEYSTROKE
        elif key == arcade.key.RIGHT or key == arcade.key.D:
            self.steer = -PLAYER_TURN_RADIANS_PER_KEYSTROKE

    def on_key_release(self, key, modifiers):
        """Called when the user releases a key. """

        if key == arcade.key.UP or key == arcade.key.W:
            self.accel = 0
        elif key == arcade.key.DOWN or key == arcade.key.S:
            self.accel = 0
        elif key == arcade.key.SPACE:
            self.brake = 0
        elif key == arcade.key.LEFT or key == arcade.key.A:
            self.steer = 0
        elif key == arcade.key.RIGHT or key == arcade.key.D:
            self.steer = 0

    def update(self, _delta_time):
        """ Movement and game logic """
        env = self.env
        for i, agent in enumerate(env.all_agents):
            sprite = self.player_list[i]
            if self.human_controlled:
                # print(f'setting throttle to {self.accel} for agent {i}')
                if env.agent_index == 0:
                    steer = self.steer
                    accel = self.accel
                    brake = self.brake
                else:
                    steer = 0
                    accel = random()
                    brake = 0

                if agent.update_intermediate_physics:
                    if agent.last_step_output == PARTIAL_PHYSICS_STEP:
                        agent.possibly_partial_step()
                    else:
                        env.step([steer, accel, brake])

                else:
                    env.step([steer, accel, brake])

                if agent.done:
                    agent.reset()

            sprite.center_x = agent.x * self.px_per_m
            sprite.center_y = agent.y * self.px_per_m
            sprite.angle = math.degrees(agent.angle)


    # def update(self, _delta_time):
    #     """ Movement and game logic """
    #     env = self.env
    #     for i, agent in enumerate(env.all_agents):
    #         sprite = self.player_list[i]
    #         if self.human_controlled:
    #             if env.agent_index == 0:
    #                 steer = self.steer
    #                 accel = self.accel
    #                 brake = self.brake
    #             else:
    #                 steer = 0
    #                 accel = random()
    #                 brake = 0
    #
    #             if self.physics_generators[i] is None:
    #                 phys_gen = env.step([steer, accel, brake])
    #                 self.physics_generators[i] = phys_gen
    #             else:
    #                 phys_gen = self.physics_generators[i]
    #             if next(phys_gen, False) is False:
    #                 self.physics_generators[i] = None
    #
    #             if agent.done:
    #                 agent.reset()
    #
    #         sprite.center_x = agent.x * self.px_per_m
    #         sprite.center_y = agent.y * self.px_per_m
    #         sprite.angle = math.degrees(agent.angle)
    #
    #     self.updates += 1

    # def update(self, _delta_time):
    #     """ Movement and game logic """
    #     env = self.env
    #     for i, agent in enumerate(env.all_agents):
    #         sprite = self.player_list[i]
    #         if self.human_controlled:
    #             if env.agent_index == 0:
    #                 steer = self.steer
    #                 accel = self.accel
    #                 brake = self.brake
    #             else:
    #                 steer = 0
    #                 accel = random()
    #                 brake = 0
    #
    #             if self.queued_states is not None:
    #                 q = self.queued_states[i]
    #                 if not q:
    #                     env.step([steer, accel, brake])
    #                     self.add_to_queued_states(agent, agent_index=i)
    #                 x, y, a = q.pop()
    #             else:
    #                 env.step([steer, accel, brake])
    #                 x = agent.x
    #                 y = agent.y
    #                 a = agent.angle
    #         else:
    #             if self.queued_states is not None:
    #                 q = self.queued_states[i]
    #                 if not q or agent.x != q[0]:
    #                     self.add_to_queued_states(agent, agent_index=i)
    #                 x, y, a = q.pop()
    #
    #             if agent.done:
    #                 agent.reset()
    #
    #         sprite.center_x = x * self.px_per_m
    #         sprite.center_y = y * self.px_per_m
    #         sprite.angle = math.degrees(a)
    #     self.updates += 1

            # log.trace(f'x:{a.x}')
            # log.trace(f'y:{a.y}')
            # log.trace(f'angle:{self.sprite.angle}')

    def add_to_queued_states(self, agent, agent_index):
        self.queued_states[agent_index].clear()
        pso = self.env.physics_steps_per_observation
        # Calc interp values between prev_x and x
        diff_x = agent.x - agent.prev_x
        diff_y = agent.y - agent.prev_y
        diff_a = agent.angle - agent.prev_angle
        for j in range(pso):
            interp = (j + 1) / pso
            self.queued_states[agent_index].append([
                agent.prev_x + interp * diff_x,
                agent.prev_y + interp * diff_y,
                agent.prev_angle + interp * diff_a])


def start(env=None, fps=60, env_config=None):
    if not env and env_config is None:
        env_config = dict(
            end_on_lane_violation='--end-on-lane-violation' in sys.argv)
    player = Deepdrive2DPlayer(
        static_obstacle='--static-obstacle' in sys.argv,
        one_waypoint='--one-waypoint-map' in sys.argv,
        is_intersection_map='--intersection' in sys.argv,
        env=env,
        fps=fps,
        env_config=env_config
    )
    player.setup()
    if 'DISABLE_GC' in os.environ:
        import gc
        log.warning('Disabling garbage collection!')
        gc.disable()

    if env is None:
        # Otherwise, env is calling the renders/blits/etc...
        arcade.run()

    return player


if __name__ == "__main__":
    start()
