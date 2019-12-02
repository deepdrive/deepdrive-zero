from gym.envs.registration import register

register(
    id='deepdrive-2d-v0',
    entry_point='deepdrive_2d.envs:Deepdrive2DEnv',
)

register(
    id='deepdrive-2d-one-waypoint-v0',
    entry_point='deepdrive_2d.envs:OneWaypointEnv',
)

register(
    id='deepdrive-2d-one-waypoint-plus-accel-v0',
    entry_point='deepdrive_2d.envs:OneWaypointPlusAccelEnv',
)
# register(
#     id='foo-extrahard-v0',
#     entry_point='gym_foo.envs:FooExtraHardEnv',
# )
