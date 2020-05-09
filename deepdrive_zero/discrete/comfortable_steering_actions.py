COMFORTABLE_STEERING_ACTIONS = {
    0: 'IDLE',  # Zero steer, zero accel

    # Based on simple heuristic solution to the single waypoint env where
    # steering set to 10% of the delta between current and desired heading
    # smoothly steers towards the waypoint. Here we expect the net to use
    # the large and small steering to comfortably set the required initial
    # angle that can then be decayed in the same way to reach the waypoint.
    1: 'DECAY_STEERING',

    # Small steering adjustments
    2: 'SMALL_STEER_LEFT',
    3: 'SMALL_STEER_RIGHT',

    # Largest comfortable left at current velocity
    4: 'LARGE_STEER_LEFT',
    5: 'LARGE_STEER_RIGHT',

    # TODO: e-steer (max steer without losing traction)
}