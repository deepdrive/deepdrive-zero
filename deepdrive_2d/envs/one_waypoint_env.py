from deepdrive_2d.envs import Deepdrive2DEnv


class OneWaypointEnv(Deepdrive2DEnv):
    def __init__(self):
        super().__init__(one_waypoint_map=True, match_angle_only=True)


class OneWaypointPlusAccelEnv(Deepdrive2DEnv):
    def __init__(self):
        super().__init__(one_waypoint_map=True, match_angle_only=False)
