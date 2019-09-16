import math
import sys
import time

from box import Box
from loguru import logger as log

import arcade
from constants import SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_MARGIN, MAP_WIDTH, \
    MAP_HEIGHT, PLAYER_TURN_RADIANS_PER_KEYSTROKE, SCREEN_TITLE, \
    CHARACTER_SCALING, PLAYER_MOVEMENT_SPEED, TESLA_LENGTH, VOYAGE_VAN_LENGTH, \
    USE_VOYAGE, VEHICLE_PNG
# Constants
from env import Environment
from map_gen import gen_map


# TODO: Calculate rectangle points and confirm corners are at same location in
#   arcade.

# TODO: Calculate lane deviation

class Spud(arcade.Window):
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
        self.env: Environment = None
        self.steer = 0
        self.accel = 0
        self.brake = False
        self.update_time = None
        self.map = None
        self.angle = None
        self.background = None
        self.meters_per_frame_speed = None
        self.rough_pixels_per_meter = None

    def setup(self):
        """ Set up the game here. Call this function to restart the game. """
        self.player_list = arcade.SpriteList()
        self.player_sprite = arcade.Sprite(VEHICLE_PNG,
                                           CHARACTER_SCALING)

        map_x, map_y = gen_map(should_save=True,
                               map_width=MAP_WIDTH,
                               map_height=MAP_HEIGHT,
                               screen_margin=SCREEN_MARGIN)
        self.map = list(zip(list(map_x), list(map_y)))

        self.background = arcade.load_texture("images/map.png")

        self.player_sprite.center_x = map_x[0]
        self.player_sprite.center_y = map_y[0]

        vehicle_length = self.player_sprite.height
        vehicle_width = self.player_sprite.width
        if USE_VOYAGE:
            vehicle_length_meters = VOYAGE_VAN_LENGTH
        else:
            vehicle_length_meters = TESLA_LENGTH
        self.rough_pixels_per_meter = vehicle_length / TESLA_LENGTH
        self.meters_per_frame_speed = \
            PLAYER_MOVEMENT_SPEED * self.rough_pixels_per_meter

        self.env = Environment(
            x=self.player_sprite.center_x,
            y=self.player_sprite.center_y,
            vehicle_width=self.player_sprite.width,
            vehicle_height=self.player_sprite.height,
            map=Box(x=map_x,
                    y=map_y,
                    arr=self.map,
                    width=MAP_WIDTH,
                    height=MAP_HEIGHT),
            add_rotational_friction=self.add_rotational_friction,
            add_longitudinal_friction=self.add_longitudinal_friction,
        )
        self.player_list.append(self.player_sprite)
        self.wall_list = arcade.SpriteList()

        # self.physics_engine = arcade.PhysicsEngineSimple(self.player_sprite,
        #                                                  self.wall_list)

    def on_draw(self):
        arcade.start_render()

        # Draw the background texture
        bg_scale = 1.1
        arcade.draw_texture_rectangle(
            MAP_WIDTH // 2 + SCREEN_MARGIN,
            MAP_HEIGHT // 2 + SCREEN_MARGIN,
            MAP_WIDTH * bg_scale,
            MAP_HEIGHT * bg_scale,
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
            self.accel = self.meters_per_frame_speed
        elif key == arcade.key.DOWN or key == arcade.key.S:
            self.accel = -self.meters_per_frame_speed
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
        if self.update_time is None:
            # init
            self.update_time = time.time()
            return

        dt = time.time() - self.update_time

        # self.bike_model.velocity += self.accel
        log.trace(f'v:{self.env.speed}')
        log.trace(f'a:{self.accel}')
        log.trace(f'dt1:{dt}')
        log.trace(f'dt2:{_delta_time}')

        obz = self.env.step(self.steer, self.accel, self.brake, dt)

        self.player_sprite.center_x = x
        self.player_sprite.center_y = y
        self.player_sprite.angle = math.degrees(angle)

        self.player_sprite.center_x = obz.x
        self.player_sprite.center_y = obz.y
        self.player_sprite.angle = math.degrees(obz.angle)

        log.trace(f'x:{obz.x}')
        log.trace(f'y:{obz.y}')
        log.trace(f'angle:{self.player_sprite.angle}')

        # TODO: Change rotation axis to rear axle (now at center)
        self.update_time = time.time()


def main():
    window = Spud(
        add_rotational_friction='--rotational-friction' in sys.argv,
        add_longitudinal_friction='--longitudinal-friction' in sys.argv,
    )
    window.setup()
    arcade.run()


def play():
    texture = arcade.load_texture("images/tesla-up.png")


if __name__ == "__main__":
    main()
