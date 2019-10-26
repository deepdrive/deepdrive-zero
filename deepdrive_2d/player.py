import math
import os
import sys
import time

from box import Box
from loguru import logger as log

import arcade
from deepdrive_2d.constants import SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_MARGIN, \
    MAP_WIDTH_PX, MAP_HEIGHT_PX, PLAYER_TURN_RADIANS_PER_KEYSTROKE, \
    SCREEN_TITLE, \
    CHARACTER_SCALING, MAX_PIXELS_PER_SEC_SQ, TESLA_LENGTH, VOYAGE_VAN_LENGTH, \
    USE_VOYAGE, VEHICLE_PNG, MAX_METERS_PER_SEC_SQ
# Constants
from deepdrive_2d.envs.env import Deepdrive2DEnv
from deepdrive_2d.map_gen import gen_map


# TODO: Calculate rectangle points and confirm corners are at same location in
#   arcade.


# noinspection PyAbstractClass
class Deepdrive2DPlayer(arcade.Window):
    def __init__(self, add_rotational_friction=False,
                 add_longitudinal_friction=False):

        # Call the parent class and set up the window
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.add_rotational_friction = add_rotational_friction
        self.add_longitudinal_friction = add_longitudinal_friction

        arcade.set_background_color(arcade.csscolor.CORNFLOWER_BLUE)
        self.player_sprite: arcade.Sprite = None
        self.player_list = None
        self.wall_list = None
        self.physics_engine = None
        self.env: Deepdrive2DEnv = None
        self.steer = 0
        self.accel = 0
        self.brake = False
        self.map = None
        self.angle = None
        self.background = None
        self.max_accel = None
        self.px_per_m = None

    def setup(self):
        """ Set up the game here. Call this function to restart the game. """
        self.player_list = arcade.SpriteList()
        self.player_sprite = arcade.Sprite(VEHICLE_PNG,
                                           CHARACTER_SCALING)

        vehicle_length_pixels = self.player_sprite.height
        vehicle_width_pixels = self.player_sprite.width
        if USE_VOYAGE:
            vehicle_length_meters = VOYAGE_VAN_LENGTH
        else:
            vehicle_length_meters = TESLA_LENGTH
        self.px_per_m = vehicle_length_pixels / vehicle_length_meters
        self.max_accel = MAX_PIXELS_PER_SEC_SQ / self.px_per_m

        width_pixels = self.player_sprite.width
        height_pixels = self.player_sprite.height

        self.env = Deepdrive2DEnv(
            vehicle_width=width_pixels / self.px_per_m,
            vehicle_height=height_pixels / self.px_per_m,
            px_per_m=self.px_per_m,
            add_rotational_friction=self.add_rotational_friction,
            add_longitudinal_friction=self.add_longitudinal_friction,
            return_observation_as_array=False,
            ignore_brake=False,
            expect_normalized_actions=False,
            decouple_step_time=True,
            physics_steps_per_observation=1,
        )
        self.env.reset()

        self.background = arcade.load_texture("images/map.png")

        self.player_sprite.center_x = self.env.map.x_pixels[0]
        self.player_sprite.center_y = self.env.map.y_pixels[0]

        self.player_list.append(self.player_sprite)
        self.wall_list = arcade.SpriteList()

        # self.physics_engine = arcade.PhysicsEngineSimple(self.player_sprite,
        #                                                  self.wall_list)

    def on_draw(self):
        arcade.start_render()

        if '--one-waypoint-map' in sys.argv:
            arcade.draw_point(self.env.map.x_pixels[1], self.env.map.y_pixels[1],
                              arcade.color.WHITE, 10)
        else:
            # Draw the background texture
            bg_scale = 1.1
            arcade.draw_texture_rectangle(
                MAP_WIDTH_PX // 2 + SCREEN_MARGIN,
                MAP_HEIGHT_PX // 2 + SCREEN_MARGIN,
                MAP_WIDTH_PX * bg_scale,
                MAP_HEIGHT_PX * bg_scale,
                self.background)

        # arcade.draw_line(300, 300, 300 + self.player_sprite.height, 300,
        #                  arcade.color.WHITE)
        # arcade.draw_lines(self.map, arcade.color.ORANGE, 3)
        # arcade.draw_point(self.heading_x, self.heading_y,
        #                   arcade.color.WHITE, 10)
        self.player_list.draw()

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """
        if key == arcade.key.UP or key == arcade.key.W:
            self.accel = MAX_METERS_PER_SEC_SQ
        elif key == arcade.key.DOWN or key == arcade.key.S:
            self.accel = -MAX_METERS_PER_SEC_SQ
        elif key == arcade.key.SPACE:
            self.brake = True
        elif key == arcade.key.LEFT or key == arcade.key.A:
            self.steer = math.pi * PLAYER_TURN_RADIANS_PER_KEYSTROKE
        elif key == arcade.key.RIGHT or key == arcade.key.D:
            self.steer = -math.pi * PLAYER_TURN_RADIANS_PER_KEYSTROKE

    def on_key_release(self, key, modifiers):
        """Called when the user releases a key. """

        if key == arcade.key.UP or key == arcade.key.W:
            self.accel = 0
        elif key == arcade.key.DOWN or key == arcade.key.S:
            self.accel = 0
        elif key == arcade.key.SPACE:
            self.brake = False
        elif key == arcade.key.LEFT or key == arcade.key.A:
            self.steer = 0
        elif key == arcade.key.RIGHT or key == arcade.key.D:
            self.steer = 0

    def update(self, _delta_time):
        """ Movement and game logic """

        # self.bike_model.velocity += self.accel
        log.trace(f'v:{self.env.speed}')
        log.trace(f'a:{self.accel}')
        log.trace(f'dt2:{_delta_time}')

        obz, reward, done, info = self.env.step(
            [self.steer, self.accel, self.brake])

        if obz:
            # log.debug(f'Deviation: '
            #           f'{obz.lane_deviation / self.rough_pixels_per_meter}')

            self.player_sprite.center_x = obz.x * self.px_per_m
            self.player_sprite.center_y = obz.y * self.px_per_m

            # TODO: Change rotation axis to rear axle (now at center)
            self.player_sprite.angle = math.degrees(obz.angle)

            log.trace(f'x:{obz.x}')
            log.trace(f'y:{obz.y}')
            log.trace(f'angle:{self.player_sprite.angle}')

        if done:
            self.env.reset()


def main():
    window = Deepdrive2DPlayer(
        add_rotational_friction='--rotational-friction' in sys.argv,
        add_longitudinal_friction='--longitudinal-friction' in sys.argv,
    )
    window.setup()
    if 'DISABLE_GC' in os.environ:
        import gc
        log.warning('Disabling garbage collection!')
        gc.disable()
    arcade.run()


def play():
    texture = arcade.load_texture("images/tesla-up.png")


if __name__ == "__main__":
    main()
