import math
import sys
import time
from math import pi, cos, sin

import arcade
from loguru import logger as log

# Constants
from bike_model import BikeModel

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

        arcade.set_background_color(arcade.csscolor.CORNFLOWER_BLUE)
        self.player_sprite: arcade.Sprite = None
        self.player_list = None
        self.wall_list = None
        self.physics_engine = None
        self.bike_model: BikeModel = None
        self.steer = 0
        self.accel = 0
        self.brake = False
        self.update_time = None
        self.add_rotational_friction = add_rotational_friction
        self.add_longitudinal_friction = add_longitudinal_friction

    def setup(self):
        """ Set up the game here. Call this function to restart the game. """
        self.player_list = arcade.SpriteList()
        self.player_sprite = arcade.Sprite("images/tesla.png",
                                           CHARACTER_SCALING)
        self.player_sprite.center_x = 64
        self.player_sprite.center_y = 120

        # TODO: Map pixels to meters
        self.bike_model = BikeModel(
            x=self.player_sprite.center_x,
            y=self.player_sprite.center_y,
            width=self.player_sprite.width,
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
        # TODO: Move the angle to the bike model

        if self.update_time is None:
            # init
            self.update_time = time.time()
            return

        dt = time.time() - self.update_time

        # self.bike_model.velocity += self.accel
        log.debug(f'v:{self.bike_model.speed}')
        log.debug(f'a:{self.accel}')
        log.debug(f'dt1:{dt}')
        log.debug(f'dt2:{_delta_time}')

        if self.brake:
            self.bike_model.speed = 0.97 * self.bike_model.speed
            # self.accel = -min(
            #     2 * self.bike_model.velocity * METERS_PER_FRAME_SPEED,
            #     math.inf * MAX_BRAKE_G * G_ACCEL * ROUGH_PIXELS_PER_METER)

        # Swap x and y to rotate back into the arcade window's frame where
        # x is right and y is straight
        change_y, change_x, yaw_rate, velocity = \
            self.bike_model.step(self.steer, self.accel, dt)

        log.debug(f'change_x:{change_x}')
        log.debug(f'change_y:{change_y}')
        log.debug(f'yaw_rate:{yaw_rate}')

        # self.player_sprite.center_x += self.bike_model.velocity

        theta1 = self.player_sprite.radians
        theta2 = theta1 - pi / 2
        world_change_x = change_x * cos(theta2) + change_y * cos(theta1)
        world_change_y = change_x * sin(theta2) + change_y * sin(theta1)
        self.player_sprite.center_x += world_change_x
        self.player_sprite.center_y += world_change_y
        self.player_sprite.angle += math.degrees(yaw_rate)

        # TODO: Change rotation axis to rear axle (now at center)

        # self.player_sprite.x = velocity * math.cos(
        #     self.player_sprite.radians)
        # self.player_sprite.change_y = velocity * math.sin(
        #     self.player_sprite.radians)
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
