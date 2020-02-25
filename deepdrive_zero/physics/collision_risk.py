"""
Collision-risk

We need to add a collision-risk penalty to the reward.
This will incentivize cars turning across traffic to yield
(among many other things). Since we know all the waypoints of the
agents in sim, we can just add collision-risk to the reward during training
without needing to input other agents' waypoints at test time.
The waypoints will be inferred at test time, which means we'll have to have
a good mix of different routes during training in order to transfer to
real driving.

I think the right units for this is just v/t, i.e m/s^2 same as accel.
Time to collision by itself does not consider the velocities and therefore
ignores the severity of the collision. Adding mass
seems practical at first,
but we don't want to say that pedestrian collisions are not
as important as vehicle collisions. The impulse for the pedestrian would be
high - but not as high as two cars colliding - so I think it's best to treat
all actors as essentially having the same mass and that any large differences
in just velocity which occur at the same location in space-time is a large risk.

To calculate this, we'll need to guess whether actors will be colocated.
In training we can use the next waypoint and calculate a velocity
range for each agent at that waypoint. The max velocity would be

v_max = v_current + v_after_max_accel_to_waypoint

Where v_after_max_accel_to_waypoint can be found with:

dist_to_waypoint = v_current * t + 1/2 * max-accel * t^2

Then after solving for t:

v_after_max_accel_to_waypoint = t * max-a

Where v is in the direction of the linear path specified by the waypoints.

Similarly the min velocity should be calculated with max-decel.

Now we can compute collision risk as Δv/Δt when there is a chance that the
actors could be colocated.

To determine colocation potential in the future - we compute the time
the front and back of each actor will be at the intersection point given max
accel and decel. So for two agents you would have the min and max time of
their front and back at the intersection point. Then if those time ranges
overlap, you return Δv/Δt as the collision risk
for max accel or decel, whichever leads to that intersection time.
This basically says, these agents could collide at this
Δv in Δt seconds - the risk is |Δv|/Δt. If max accel and decel could both
lead to collisions, use the max |Δv|/Δt.

This means that, for example, even after an oncoming car has passed you,
and say you're trying to turn left across that car's lane,
there is a risk the other car could max-decel,
go into reverse, and collide with you.
This may seem silly, but actually is not if you
consider a car slowly passing through the intersection due to waiting for a
ped to cross, or because traffic is just backed up.
So basically collision risk
determines if the agents paths intersect,
and if so, weight the collision risk by the velocity difference and
how quickly it could happen.

A problem with this is that vehicles whose linearly connected waypoint
paths don't intersect, could still collide, if the agent moves out of its
lane for example. Lane keeping reward and collision penalty already
disincentivize this, but intersecting another vehicle's path should be an issue.

Also, for the vehicle coming straight, we need to basically say, this is okay...!

Perhaps the risk should be simple, just say if you're making a left in RHT,
or a right in LHT, wait until all oncoming vehicles have passed. Now
we're getting into behavior type rules but the agent will also make
seamless tradeoffs between this penalty and collision penalty say if the
other agent starts to come towards us, we will be able to seamlessly
make a tradeoff and go into the intersection to avoid an accident even if
that means not waiting until the other agent is past...

Shear is somewhat taken into consideration by computing the velocity vector
difference, but this could be expanded on. Also impact location is important,
i.e. T-bone crashes are more deadly than head on, so that should be
taken into consideration as well.

"""
