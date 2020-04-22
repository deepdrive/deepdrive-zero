class PhysicsInterpolationState:
    def __init__(self, total_steps):
        self.i = 0
        self.total_steps = total_steps
        # self.curr_steer = None
        # self.curr_accel = None
        # self.curr_brake = None
        # self.interp_steer = None
        # self.interp_accel = None
        # self.interp_brake = None

    def ready(self):
        return self.i == 0

    def update(self,
               # curr_steer,
               # curr_accel,
               # curr_brake,
               # interp_steer,
               # interp_accel,
               # interp_brake
               ):
        self.i += 1
        self.i = self.i % self.total_steps
        # self.curr_steer = curr_steer
        # self.curr_accel = curr_accel
        # self.curr_brake = curr_brake
        # self.interp_steer = interp_steer
        # self.interp_accel = interp_accel
        # self.interp_brake = interp_brake
