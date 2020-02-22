Code complete: Save model when new max episode return avg or max, horizon return avg or max, 
or trip completion max achieved on epoch

Add left yield road rule reward (eventually need against-traffic-turn input 
signal i.e. turn left for right-hand-traffic and turn right for 
left-hand-traffic - )

Do curriculum again
    First no g penalty - no jerk penalty - just destination and collision.
    Then add high g penalty + jerk penalty and perhaps reduce gamma to 0.9 (1 second lookahead) or 0.95 (2 second lookahead) 

Also try lower lr and gamma

Try LSTM / GTrXL



Also try multi timescale net a la deep mind ctf 

Fix mpi training - somehow agent performance is much worse when num_cpu > 1. 
Could be due to my multi-agent setup somehow being flawed, a problem in the
original spinningup, or something else. Also just try num_cpu=1 to see
if the backgrounding somehow hurts things. It's also obvious they didn't
try num_cpu with tensorflow as they don't limit the GPU memory tensorflow takes
and setting num_cpu=2 will crash immediately.

We need to take many agents actions as input into the env, instead of 
just one. Especially, we need to compute physics for each agent more 
concurrently than we do now. Ideally, it would be a parallel numba calc
that computed all agents positions at once. Collisions could then be computed
at the physics tic rate (currently 60fps) during every physics loop step.

Make sure we can take drastic action to avoid collision even with action/gforce penalties

Penalize the complexity of a course of action
- Try squaring g-force/action penalty like hopper and LQR
- Penalize the instantaneous second derivative of the action, i.e. rate changes across two frames (squared). So if the steering change in frame 1 is -0.1 (going from 0.2 to 0.1) and in frame 2 is 0.1 (going from 0.1 back to 0.2), then the instantaneous second derivative is -0.1 - 0.1 = -0.2. However, if we instead went to 0 in the second frame the second derivative would be -0.1 - -0.1 = 0. Instead of steering, a finite difference approximation, using the same method as above, to the g-force derivative should be used in order to account for speed. This is also known as jerk. https://en.wikipedia.org/wiki/Jerk_(physics). In order to account for multiple frames, we could average jerk over the past second by storing 10 frames of g-force and g-force change, averaging it, and passing it to the network.


Don't overwrite saved models.

Make sure seeds determine all randomness and that we have complete determinism per seed

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
