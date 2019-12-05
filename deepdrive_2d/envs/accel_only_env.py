from deepdrive_2d.envs import Deepdrive2DEnv


class AccelOnlyEnv(Deepdrive2DEnv):
    def __init__(self):
        super().__init__(one_waypoint_map=True, accel_only=True)
