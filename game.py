import math
import sys
import time

from box import Box
from loguru import logger as log

import arcade
from map_gen import gen_map
# Constants
from world_model import Dynamics

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCREEN_MARGIN = 50
MAP_WIDTH = SCREEN_WIDTH - SCREEN_MARGIN * 2
MAP_HEIGHT = SCREEN_HEIGHT - SCREEN_MARGIN * 2
SCREEN_TITLE = 'spud'  # self-play unreal driving?

# TODO: Move these into instance properties
CHARACTER_SCALING = 1/4
PLAYER_MOVEMENT_SPEED = 5  # pixels per frame Pacifica Hybrid
# PLAYER_MOVEMENT_SPEED = 10  # pixels per frame Model 3
VEHICLE_LENGTH = 320
VEHICLE_WIDTH = 217
ROUGH_PIXELS_PER_METER = VEHICLE_LENGTH / 4.694
METERS_PER_FRAME_SPEED = PLAYER_MOVEMENT_SPEED * ROUGH_PIXELS_PER_METER


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
        self.dynamics: Dynamics = None
        self.steer = 0
        self.accel = 0
        self.brake = False
        self.update_time = None
        self.map = None
        self.angle = None

    def setup(self):
        """ Set up the game here. Call this function to restart the game. """
        self.player_list = arcade.SpriteList()
        self.player_sprite = arcade.Sprite("images/tesla-up.png",
                                           CHARACTER_SCALING)

        map_x, map_y = gen_map(should_save=True,
                               map_width=MAP_WIDTH,
                               map_height=MAP_HEIGHT,
                               screen_margin=SCREEN_MARGIN)
        self.map = list(zip(list(map_x), list(map_y)))

        self.player_sprite.center_x = map_screen_x[0]
        self.player_sprite.center_y = map_screen_y[0]

        self.player_sprite.center_x = map_x[0]
        self.player_sprite.center_y = map_y[0]

        self.dynamics = Dynamics(
            x=self.player_sprite.center_x,
            y=self.player_sprite.center_y,
            width=self.player_sprite.width,
            height=self.player_sprite.height,
            map=Box(x=map_x,
                    y=map_y,
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

        # TODO: Create a background image as this takes 97% CPU time and reduces
        #  FPS from 60 to 20 and sometimes ~1 FPS.
        arcade.draw_lines(self.map, arcade.color.ORANGE, 3)
        arcade.draw_point(self.heading_x, self.heading_y,
                          arcade.color.WHITE, 10)
        self.player_list.draw()

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """
        if key == arcade.key.UP or key == arcade.key.W:
            self.accel = METERS_PER_FRAME_SPEED
        elif key == arcade.key.DOWN or key == arcade.key.S:
            self.accel = -METERS_PER_FRAME_SPEED
        elif key == arcade.key.SPACE:
            self.brake = True
        elif key == arcade.key.LEFT or key == arcade.key.A:
            self.steer = math.pi / 16
        elif key == arcade.key.RIGHT or key == arcade.key.D:
            self.steer = -math.pi / 16

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
        log.trace(f'v:{self.dynamics.speed}')
        log.trace(f'a:{self.accel}')
        log.debug(f'dt1:{dt}')
        log.trace(f'dt2:{_delta_time}')

        x, y, angle = self.dynamics.step(self.steer,
                                         self.accel, self.brake, dt)

        self.player_sprite.center_x = x
        self.player_sprite.center_y = y
        self.player_sprite.angle = math.degrees(angle)

        log.trace(f'x:{x}')
        log.trace(f'y:{y}')
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


if __name__ == "__main__":
    main()
