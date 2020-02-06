Rename to deepdrive-0, deepdrive-zero? - We will want to add 3D eventually, but the main advantage of this env is that it doesn't depend on Unreal or require Linux / GPUs - with all the physics in numba/python making it accessible, fast, and tailor made for the self-driving problem. The Unreal based sim will be the ultimate test for agents trained here, but is likely too heavyweight for fast iteration and scaled up training.

Create multi-agent unprotected left, 2 agents A and B, A is turning left B is coming towards A in opposing lane
 - First waypoint for agent A will be inside intersection, second will be after left
 - Waypoint for agent B will be past intersection going straight
 - A must yield to B if time to collision under threshold
 - Lane boundaries? Eventually we will need lane widths for waypoints and start positions. If we avoid this now, agent may do something crazy. We could import some map. We could put static obstacles on edges of lanes, but that doesn't work for intersection.
 

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
