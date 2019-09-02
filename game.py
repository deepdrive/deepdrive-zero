import math
import sys
import time

from loguru import logger as log

import arcade
# Constants
from bike_model import VehicleDynamics

# TODO: Calculate rectangle points and confirm corners are at same location in
#   arcade.

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 650
SCREEN_TITLE = 'spud'  # self-play unreal driving?

CHARACTER_SCALING = 1/4
PLAYER_MOVEMENT_SPEED = 5  # pixels per frame
ROUGH_PIXELS_PER_METER = 20
METERS_PER_FRAME_SPEED = PLAYER_MOVEMENT_SPEED * ROUGH_PIXELS_PER_METER


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
        self.vehicle_dynamics: VehicleDynamics = None
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
        self.player_sprite.center_x = 64
        self.player_sprite.center_y = 120

        # TODO: Map pixels to meters
        self.vehicle_dynamics = VehicleDynamics(
            x=self.player_sprite.center_x,
            y=self.player_sprite.center_y,
            width=self.player_sprite.width,
            height=self.player_sprite.height,
            angle=self.player_sprite.angle,
            add_rotational_friction=self.add_rotational_friction,
            add_longitudinal_friction=self.add_longitudinal_friction,
        )
        self.player_list.append(self.player_sprite)
        self.wall_list = arcade.SpriteList()

        # self.physics_engine = arcade.PhysicsEngineSimple(self.player_sprite,
        #                                                  self.wall_list)

    def on_draw(self):
        arcade.start_render()
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

    # TODO: njit this
    def update(self, _delta_time):
        """ Movement and game logic """
        if self.update_time is None:
            # init
            self.update_time = time.time()
            return

        dt = time.time() - self.update_time

        # self.bike_model.velocity += self.accel
        log.trace(f'v:{self.vehicle_dynamics.speed}')
        log.trace(f'a:{self.accel}')
        log.trace(f'dt1:{dt}')
        log.trace(f'dt2:{_delta_time}')

        x, y, angle = self.vehicle_dynamics.step(self.steer,
                                                 self.accel, self.brake, dt)

        self.player_sprite.center_x = x
        self.player_sprite.center_y = y
        self.player_sprite.angle = math.degrees(angle)

        log.trace(f'x:{x}')
        log.trace(f'y:{y}')
        log.trace(f'angle:{angle}')

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
