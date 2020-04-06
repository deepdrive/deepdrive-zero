Try lowering learning rate on unimproving models.

Try resuming without restoring Adam rates
 
Try penalizing only g-force > 1, keep jerk same.

Try mixed training. Sometimes speed only, sometimes speed + comfort.

Try adding another MLP which connects to the same outputs with weights preferring
the old MLP such that some knowledge gets distilled in the new net, but with
more plasticity to learn new techniques.

[Did not work] Try game over on exiting lane from scratch + fine-tune.

Try higher lane penalties and move penalties from scratch + fine-tune.

Try end on harmful g from scratch + fine-tune.

Try training from scratch with yield, collision, and speed => fine tune comfort

Change the step rate to 5 steps per second (200ms), as human reaction time is 
about this much and this should help learning efficiency, plus it will make constrained
controls less of an issue. 

Filter fixed size of other actors to feed into network using distance and worst
case TTC (i.e. if they changed position). Could also do max-pool, but don't understand
well how that works in OpenAI V and if that can apply to many more agents. Also transformer
or LSTM could work that scans all actors where a higher number of actors results
in effectively higher latency encodings of actors fed to MLP.

Add braking+accel negative reward that penalizes pressing brake and accel at the 
same time.

Add left yield road rule reward (eventually need against-traffic-turn input 
to net i.e. turn left for right-hand-traffic and turn right for 
left-hand-traffic - right now can be gleaned from other inputs like waypoint 
angle)

Investigate if the gradient of the gaussian likliehood wrt the prob:
https://www.wolframalpha.com/input/?i=d%2Fds+1%2F%28s*%282*pi%29**0.5%29+*+exp%28-0.5%28%28x-u%29%2F%28s%29%29%29+
Is causing large std gradient for low probs and thus dominating the clipping range every step.
https://www.google.com/search?sxsrf=ALeKk00H0iDI57_JildOxjswi-9G03TPxQ%3A1584296060686&ei=fHBuXoK6KYz7-gT7k6sY&q=exp%281%2Fx%29*x%2Fx**3&oq=exp%281%2Fx%29*x%2Fx**3&gs_l=psy-ab.3...190972.208594..209234...2.1..0.79.1173.17......0....1..gws-wiz.......0i71j35i39j0i131i273j0i273j0j0i131j0i67j0i30j0i8i30j0i5i30.o4HAS4RVVxA&ved=0ahUKEwiC6Knfip3oAhWMvZ4KHfvJCgMQ4dUDCAs&uact=5

To get around difficulty RL is having with constrained controls,
allow agent to fully control steering angle BUT not instantly. i.e. if it
gives a super large steering angle, it may have to wait a few timesteps. 
This will mean decoupling the agents' physics.

Store a snapshot of all python files and current git commit of env and spinningup
for every experiment. (compress)

RlPyt
- Discretize actions
- Remove Conv layers
- possibly tune hyperparams as they crushed Sota on atari but not dmlab
- create a single agent intersection env where the straight agent just does random throttle
- see if we can reduce the massive memory requirements, buffer sizes, etc... 

Try squaring rewards to avoid saturation of tanh units

Track stats for each agent individually

Fix trip percentage

Try larger batch size, lower learning rate, larger models to get past previous
intersection performance

Idea: Hierarchical RL (try recurrent model first) 
Have one model that outputs 1 second feasible destinations + velocities, trained with 
teleportation. Teleportation model will also need to be able to deal with
the prev destination not being reached  (due to some obstacle)
and we are asked for a new destination in less than 1 second. Reward would
not include g force or jerk, but would just be time, lane, yield-left, and collision.
Then use that model's destinations as rewards for a 100ms model (including penalizing 
collisions that end up diverting from above model's dest).
Create different granularities as needed. (First see if LSTM-A2C or R2D1 work though)
100ms model could be MPC as well.

Dockerize

Try 5 or 6 seeds per run

Learn what to do in anomalous situations. Horn output can be something you 
train where, honking is input to nearby actors and they learn from experience, 
say of scenarios where you send a random car through the scene, 
that when the honk input is active they should slow down / be more cautious of 
other actors around them.

Meta-learning/self-supervision idea:
In order to better explore reward space and be amenable to different styles
of driving, we should let the model explore reward weights
that achieve a desired rate of learning (not too big lest we
forget, and not too small lest we become sample inefficient, overfit, 
and/or not learn enough).
-------------------------------------------------------------------------------
Simple first pass:
If recent reward increase has flattened (or diverged), then (optionally reset
weights to a better net and) shift objective weights. 
Then if too sharp a decrease in reward, revert, else shift more. 
Find the shift that gives the desired return increase rate (DRI). 

The DRI could be dynamic by evaluating on a statically weighted objective a la 
deepdrive’s leaderboard eval where the DRI
is annealed as we reach theoretical maximum eval performance.
This is similar in spirit to population based training, but explores goal 
weightings instead of hyperparams and therefore is a bit meta-learning-ish.
-------------------------------------------------------------------------------

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
at the physics tick rate (currently 60fps) during every physics loop step.
Also we should likely output n observations for n agents, so that modifying
other RL algos can be done in a standard way where you take observations[0]
for single agent mode, and store observations[1...n] for multi-agent mode.
This will also allow taking advantage of mini-batching for computing several agents'
actions.

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
