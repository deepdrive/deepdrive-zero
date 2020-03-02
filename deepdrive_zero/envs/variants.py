from deepdrive_zero.envs import Deepdrive2DEnv


class OneWaypointSteerOnlyEnv(Deepdrive2DEnv):
    def __init__(self):
        super().__init__(is_one_waypoint_map=True, match_angle_only=True)


class OneWaypointEnv(Deepdrive2DEnv):
    def __init__(self):
        super().__init__(is_one_waypoint_map=True, match_angle_only=False)


class IncentArrivalEnv(Deepdrive2DEnv):
    def __init__(self):
        super().__init__(is_one_waypoint_map=True, match_angle_only=False,
                         incent_win=True)


class StaticObstacleEnv(Deepdrive2DEnv):
    def __init__(self):
        super().__init__(is_one_waypoint_map=True, match_angle_only=False,
                         incent_win=True, add_static_obstacle=True)

class NoGforcePenaltyEnv(Deepdrive2DEnv):
    def __init__(self):
        super().__init__(is_one_waypoint_map=True, match_angle_only=False,
                         incent_win=True, add_static_obstacle=True,
                         disable_gforce_penalty=True)


class SixtyFpsEnv(Deepdrive2DEnv):
    def __init__(self):
        super().__init__(is_one_waypoint_map=True, match_angle_only=False,
                         incent_win=True, add_static_obstacle=True,
                         disable_gforce_penalty=True,
                         physics_steps_per_observation=1)


class IntersectionEnv(Deepdrive2DEnv):
    def __init__(self):
        super().__init__(is_intersection_map=True, match_angle_only=False,
                         incent_win=True, disable_gforce_penalty=True)

class IntersectionWithGsEnv(Deepdrive2DEnv):
    def __init__(self):
        super().__init__(is_intersection_map=True, match_angle_only=False,
                         incent_win=True)

class IntersectionWithGsAllowDecelEnv(Deepdrive2DEnv):
    def __init__(self):
        super().__init__(is_intersection_map=True, match_angle_only=False,
                         incent_win=True, forbid_deceleration=False)