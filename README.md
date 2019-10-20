# Deepdrive 2D

_2D environment for self-driving_

The purpose of this environment is to create a useful "toy" environment for
self-driving that allows quickly working out bugs in deep RL algorithms before
transferring them to a 3D simulator. 

Bike model

Most industrial self-driving control has been done in 2D. 
The model humans use to drive seems _mostly_ to omit the possibility
of things moving up or down, where it is mostly 
made up of an internal map of what's around you within some radius, 
and predictions of what will happen in that
radius for around 10 seconds.

This could mean that the simple 2D model may serve as an important component
of a 3D driving system, which allows using model based RL. Again, this would
be similar to current control systems which used model based methods like
model predictive control, usually in 2D.

This decomposes the driving problem into two main parts: perception and control, 
down from ~5: perception, prediction, decision making, planning, and control.
Here we assume perception entails localization and tracking as well.

The problem with this is that if there are certain phenomenon that require 
end-to-end learning from say pixels to steering, the two part system will
not work. For example, reading a cyclist's face to determine whether they will
 go. I'm assuming here that we can postpone this in two ways

1. Add an end-to-end channel in addition to the perception / control system
for emergency situations as with Voyage's collision mitigation system.

2. Glean which parts of driving are toughest for RL, i.e. non-wobbly steering
and figure out how to solve those in 2D before graduating to 3D end-to-end
systems in a partially observed (projected) observation space. We can also use
2D to quickly test partially observed observation spaces in order to see
what recurrent methods work best in this domain, i.e RNN's, transformers, etc... 


Plan

We’ll start out with a fully observed radius around a 2D kinematic model,
so no RNN/attention *should* be needed given a large enough radius. 

Some steps along the way should be:

1. Learn non-wobbly driving which RL is notoriously prone to. 
(Look at steering values - not just trajectory) 
Perhaps the reward needs to be tuned for this - esp g-force.

2. Learn behaviors which require trading off lane deviation for speed and or 
g-force, like leaving lane to pass or avoid an obstacle. 


Architecture

The Arcade package is used for rendering, but is decoupled from physics so as
to allow running the sim quickly without rendering. We use Numba to speed
up physics calculations which allows iterating in Python (with the option
to do ahead of time compilation on predefined numeric types) while maintaining
the performance of a compiled binary. 

## EXPERIMENTAL NOTEBOOK
* Spinning up SAC was not able to accurately match (80%) even a single input, whereas pytorch SAC is able to get 99.68% at 10k steps and 99.14% at 240 steps, fluctuating in between.
* Spinning up SAC does much better at matching input when entropy (alpha) is zero, going from 65% to 80%.
Pytorch SAC does best when auto entropy tuning is on (which spinning up does not have) - you really don’t want any entropy in the loss when matching a single input as no generalization is required. However, I think entropy tuning *may* help counter the stochasticity of the policy in some weird way, as it is learned jointly with the mean and variance of the policy - and could be working together to achieve optimality.
* Matching a corrective action (essentially reversing and scaling the input) converges more slowly to 98.25% at 1840 steps (vs 240 steps for matching to get to 99.14%) than just matching the input, achieving 98.96% at 10k steps
Passing in the previous action in stable tracking helps achieve 1% higher performance from 97% to 98% and (I believe) much more quickly
* dd2d.a Something is very wrong with the critic losses, varying between 10k and 166M on critic 1 and between 100k and 2.35B on critic 2.  This coincided with a rise in the entropy temperature to 40 from zero and brought back down when the entropy temp came down to around 2.k. So the critics are just giving poor value approximations to the policy which thinks it's doing amazing - and the critics are somehow being negatively affected by the entropy tuning despite entropy tuning leading to better results in the gym-match-continuous envs. It may just be that there is nothing to learn, like feeding noise for targets and so the thing is flailing around looking for something that works. So perhaps there's missing info in the observation.
* dd2d.b Giving a speed only reward when above 2m/s and within lane deviation of 1, results in well behaved critic and policy losses. We diverge towards the end of training, but mostly due to reduced variance and get 562/~900 total possible reward. However, not much distance is accrued as the agent learns that turning fast gives it more immediate speed reward and therefore drives off the track every episode.
* dd2d.c Making the reward dependent on distance advancing along route and equal to speed vs gating it with lane_deviation and speed is working *much* better (note we are still on a static map). Not quite sure why this is, perhaps because I didn’t require a minimum speed like before. However, this diverged when the agent started to prefer high speed rewards again over staying on the track for longer.
* dd2d.d In the next experiment, I tried the same distance advance gating, but with zero or one reward. This learned more slowly, but did not diverge. It seems like we should try giving continuous reward that is confined by giving a max of 1 at the max desired speed. I tried this before and it did terribly, but it was gated by lane deviation and minimum speed. However, giving rewards of just 0 or 1, or -1 seems to be a thing in Rl.
* So funny thing, the distance based reward didn't take into account whether waypoints on the route had been visited before. So one of the agents learned to go in circles (reverse is disabled) and hit the same way point over and over, getting thousands of points when the max should have been 175! Another run with the same reward however has done really well, getting half way up the track (75 meters).
* dd2d.e So now I'm keeping track of the max distance achieved to prevent the circle cheat, and early training was good reaching 3m by 100k steps, better than 2.3 achieved in the best bot so far, but we then diverge and start to drive extremely slow. With a gamma of 0.99 - we have roughly 100 steps of future reward that we care about. At 60FPS realtime (wall time is faster), we only care about 2-3 seconds ahead. This seems fine as we receive one point roughly every half meter, so maximizing 100 step return should lead to faster progress along the route.
* Aside: training steps per second is ~20 or 1.629e6 steps in 22 hours.
* Env steps per second is 2557: 
    ```
    import deepdrive_2d
    env = gym.make('deepdrive-2d-v0')
    env.reset()
    %time env.step([0.01, 1, 0])
    CPU times: user 391 µs, sys: 37 µs, total: 428 µs
    Wall time: 445 µs
    ```
* 10 actions per second - with 60 physics steps per second allows for 2.5x-5xs faster learning. 75 meters in 1.5M steps for 60aps while in 600k steps for 10aps.
* Large batches just learning slowly, so we could do larger learning rate
* NaN's in the losses and accel after achieving most of route - not sure why - could be that our finite horizon return of 100-500 steps decreases as the episode will be over. Perhaps then, we should give 1 large reward at the end of the route. Right now, each waypoint is 0.5 meters and gives 1 reward. So we are incentivized at 10FPS to go 0.5 meters every 100ms or 5m/s or 10mph. We end up going twice that pretty consistently... so maybe I'm off by 2 somewhere. 
* Batch size of 1024, lr 0.001 converges 1.5x slower (8.3k[5:37h] test steps vs 5.6k[2:45h]) than default batch 256, lr 0.0003 - single run. Larger batches, 65k/0.002 and 32k/0.001 did not converge in 1 - 2 days of training.
* So we've added minimizing gforce to the objective of achieving waypoint rewards which has been troublesome in the past resulting in the car not going anywhere as the initial acceleration is disincented. So this time, I used a curriculum of speed first, gforce later, which seemed to work. The way I added it was to randomly give gforce penalties proportional to the avg trip pct completed by the agent (I used all time avg, so using a rolling avg would be a good change in the future). So if the agent was completing 1% of the trip on average, and performed a high g turn or deceleration, there'd be a 1% chance of losing either half or all of the frame's distance reward for 0.1g and 0.4g respectively - and for 1g, there'd be a 1% chance of a -1 reward and game over. In this way, I kept a discrete reward which is the only type that worked for distance, while combining two objectives.
* In another experiment, I started including gforce rewards at 50% avg trip completion, but 50% seemed to be too late and showed little signs of minimizing gforce before the policy and critics exploded at 800k episodes. In the experiment above, 2M episodes were completed without exploding gradients.
* Also giving gforce penalties from the start of training results in achieving zero distance after several hours of training.
* After lengthening training time with curriculum gforce (and avoiding exploding gradients with gradient clipping), I was able to get one agent out of three to decrease gforce levels while marginally increasing or maintaining distance.
