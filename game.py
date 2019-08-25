import math
from math import pi, cos, sin

import arcade

# Constants
from bike_model import BikeModel

from loguru import logger as log

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 650
SCREEN_TITLE = 'spud'  # self-play unreal driving?

CHARACTER_SCALING = 1/4
PLAYER_MOVEMENT_SPEED = 5  # pixels per frame


class Spud(arcade.Window):
    def __init__(self):

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
            self.accel = PLAYER_MOVEMENT_SPEED
        elif key == arcade.key.DOWN or key == arcade.key.S:
            self.accel = -PLAYER_MOVEMENT_SPEED
        elif key == arcade.key.SPACE:
            self.brake = True
        elif key == arcade.key.LEFT or key == arcade.key.A:
            self.steer = math.pi / 4
        elif key == arcade.key.RIGHT or key == arcade.key.D:
            self.steer = -math.pi / 4

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
    def update(self, delta_time):
        """ Movement and game logic """
        # TODO: Move the angle to the bike model

        # self.bike_model.velocity += self.accel
        log.trace(f'v:{self.bike_model.velocity}')
        log.trace(f'a:{self.accel}')

        if self.brake:
            self.accel = -2 * self.bike_model.velocity

        change_x, change_y, yaw_rate, velocity = \
            self.bike_model.step(self.steer, self.accel, delta_time)


        self.player_sprite.center_x += self.bike_model.velocity


        # theta1 = self.player_sprite.radians
        # theta2 = theta1 - pi / 2
        # world_change_x = change_x * cos(theta2) + change_y * cos(theta1)
        # world_change_y = change_x * sin(theta2) + change_y * sin(theta1)
        # self.player_sprite.change_x = world_change_x
        # self.player_sprite.center_y = world_change_y
        # self.player_sprite.angle += (math.degrees(yaw_rate) / delta_time)


        # self.player_sprite.x = velocity * math.cos(
        #     self.player_sprite.radians)
        # self.player_sprite.change_y = velocity * math.sin(
        #     self.player_sprite.radians)


def main():
    window = Spud()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
