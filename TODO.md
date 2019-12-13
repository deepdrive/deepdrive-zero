Make sure we can take drastic action to avoid collision even with action/gforce penalties

Penalize the complexity of a course of action

Try squaring action penalty like hopper and LQR

Static obstacle with some sort of collision detection

Don't overwrite saved models.

Measure gforce due to inaccurate steering

Measure gforce due to accurate steering

Try automatic entropy tuning with simple-steer

Try gforce penalties with simple-steer

Add n-step returns to prevent divergence

Make sure seeds determine all randomness and that we have complete determinism per seed

Only give distance rewards every x meters for single waypoint map.

Add an action duration, i.e. number of steps to repeat action (needs interrupt tho)

Repeat previous action inputs to have more than one weight, a la CONV kernels. (Could give more opportunities to learn important input).


Try to make initial trajectories in curriculum IID by distributing training across curvatures and initial velocities.
The first stage of this curriculum should be a turn that encompasses the agents horizon per gamma, so 100 steps at 10APS would be a 10 second episode.

Write gforce to tfx, to get an idea of average instead of just max.

Do gradient clipping using std deviation of recent 100 gradient norms instead of hardcoding at 100.

Max gforce of 200 seems wrong, maybe
during a GC or something IDK - or perhaps it's a problem with reset.


Randomly shorten map to 50-100% of current to reduce overfitting

Save model versions in case of Nan explosion, we explode after having converged for a while



Try to fill GPU by increasing batch size / learning rate (learning rate needs to be higher than 0.01 for batch 1024)



Allow replaying episode x,y's in arcade

Figure out why resume seems to be diverging and starts less performant
 - Adam internals (momentum, rmsprop) (save optimizer)
 - Start steps?
 - Replay memory? But we don't diverge right away.
 - torch.manual_seed?
 - call .eval() after load so we're not in train mode?

Add multiple agents to environment (going opposite directions?)

Add growing diversity of agents to many agent environment with intersections

Add sign to lane deviation to signal right left. Angle ahead prob accounts for this already though.

Try random maps every episode

Try making the distance rewards more granular, right now at slow speeds, they
are very sparse. (Maybe distance to next way point). There still need to be a 
threshold distance achieved so that speed is rewarded. 

1) Turn off entropy tuning or make it less aggressive
