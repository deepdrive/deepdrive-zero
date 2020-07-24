# Deepdrive Zero alpha

<p align="center">
    <img src="https://user-images.githubusercontent.com/181225/88427805-8226c480-cda8-11ea-916a-35b6685209c5.gif"><br/>
    <i>Example intersection agents trained with PPO. Details at <a href="https://smooth.deepdrive.io">smooth.deepdrive.io</a></i>
</p>

The purpose of this environment is to create a useful, extremely fast simulation environment for self-driving that allows quickly working out bugs in deep RL algorithms before transferring them to a [3D simulator](https://github.com/deepdrive/deepdrive). There's no dependency on Unreal, Linux, or GPU's - with all the physics in numba/python making it accessible, fast, and tailor made for the self-driving problem. For example, we currently end the episode on collision, avoiding the need for more complex collision handling usually used in games or simulations where the results of many collisions are of more interest than how to avoid collisions altogether as in the case of self-driving. The Unreal based sim will be the ultimate test for agents trained here, but is likely too heavyweight for fast iteration and scaled up training.

### Try it out

#### Install

Install our fork of [spinning up](https://github.com/deepdrive/spinningup)

```
git clone https://github.com/deepdrive/deepdrive-2d
pip install -e .
```

### Docker

Pull
```
docker pull deepdriveio/deepdrive_zero
```

Build container with your *committed* changes

```
export SPINNINGUP_DIR=/your/spinninupdir
make
```

Run an experiment example

```
docker run -d --name intersection_2_agents_fine_tune_add_left_yield_from_scratch deepdriveio/deepdrive_zero python /workspace/deepdrive-zero/deepdrive_zero/experiments/intersection_2_agents_fine_tune_add_left_yield_from_scratch.py
```


#### Play

You can play this sim like any other game. 

Unprotected left scenario

```
player.py --intersection --no-timeout
```

#### Train



#### Test



More examples from which you can use the SCRIPT_NAME and PARAMETERS to see what commands I'm running to do things. Or if you use PyCharm, you can use these directly.

https://github.com/deepdrive/deepdrive-2d/blob/master/.idea/runConfigurations/play_dd2d_all_friction_one_waypoint_static_obstacle_DISABLE_GAME_OVER_1.xml

### Action Space

The current action space is continuous from -1 to 1 for each policy network output. The order is 
```
steer, accel, brake
```

Negative accel can be used to put the car into reverse. Network outputs are scaled to represent physically realistic values:

**steer**
> Heading angle of the ego

**accel**
> m/s/s of the ego, positive for forward, negative for reverse

**brake**
> From 0g at -1 to 1g at 1 of brake force


View progress example

```
docker logs intersection_2_agents_fine_tune_add_left_yield_from_scratch  2>&1 | grep AverageEpRet
```

Copy data out of docker example
```
docker cp intersection_2_agents_fine_tune_add_left_yield_from_scratch:/workspace/spinningup/data /tmp/dd0-data4
```

Copy data from gcp example
```
gcloud compute scp --recurse deepdrive-zero2:/tmp/dd0-data4 ~/dd0-data/snaphot1
```
Where `deepdrive-zero2` is the name of the instance.



### Bike Model

Most industrial self-driving control has been done in 2D. The model humans use to drive seems _mostly_ to omit the possibility
of things moving up or down, where it is mostly made up of an internal map of what's around you within some radius, and predictions of what will happen in that radius for around 10 seconds.

This could mean that the simple 2D model may serve as an important component of a 3D driving system, which allows using model based RL. Again, this would be similar to current control systems which used model based methods like model predictive control, usually in 2D.

This decomposes the driving problem into two main parts: perception and control, down from ~5: perception, prediction, decision making, planning, and control. Here we assume perception entails localization and tracking as well.

The problem with this is that if there are certain phenomenon that require end-to-end learning from say pixels to steering, the two part system will not work. For example, reading a cyclist's face to determine whether they will go. I'm assuming here that we can postpone this in two ways:

1. Add an end-to-end channel in addition to the perception/control system for emergency situations as with Voyage's collision mitigation system.

2. Glean which parts of driving are toughest for RL, i.e. non-wobbly steering and figure out how to solve those in 2D before graduating to 3D end-to-end systems in a partially observed (projected) observation space. We can also use 2D to quickly test partially observed observation spaces in order to see what recurrent methods work best in this domain, i.e RNN's, transformers, etc.

### Plan

We’ll start out with a fully observed radius around a 2D kinematic model, so no RNN/attention *should* be needed given a large enough radius.

Some steps along the way should be:

1. Learn non-wobbly driving which RL is notoriously prone to. Perhaps the reward needs to be tuned for this - especially g-force.

2. Learn behaviors which require trading off lane deviation for speed and or g-force, like leaving lane to pass or avoid an obstacle.

### Architecture

The Arcade package is used for rendering, but is decoupled from physics so as to allow running the sim quickly without rendering. We use Numba to speed up physics calculations which allows iterating in Python (with the option to do ahead of time compilation on predefined numeric types) while maintaining the performance of a compiled binary.

### Bibtex

## Citing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3871907.svg)](https://doi.org/10.5281/zenodo.3871907)

Bibtex
```
@software{craig_quiter_2020_3871907,
  author       = {Craig Quiter},
  title        = {Deepdrive Zero},
  month        = jun,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {alpha},
  doi          = {10.5281/zenodo.3871907},
  url          = {https://doi.org/10.5281/zenodo.3871907}
}
```

### Experimental Notebook

* Spinning up SAC was not able to accurately match (80%) even a single input, whereas pytorch SAC is able to get 99.68% at 10k steps and 99.14% at 240 steps, fluctuating in between.

* Spinning up SAC does much better at matching input when entropy (alpha) is zero, going from 65% to 80%.
Pytorch SAC does best when auto entropy tuning is on (which spinning up does not have) - you really don’t want any entropy in the loss when matching a single input as no generalization is required. However, I think entropy tuning *may* help counter the stochasticity of the policy in some weird way, as it is learned jointly with the mean and variance of the policy - and could be working together to achieve optimality.

* Matching a corrective action (essentially reversing and scaling the input) converges more slowly to 98.25% at 1840 steps (vs 240 steps for matching to get to 99.14%) than just matching the input, achieving 98.96% at 10k steps. Passing in the previous action in stable tracking helps achieve 1% higher performance from 97% to 98% and (I believe) much more quickly.

* dd2d.a Something is very wrong with the critic losses, varying between 10k and 166M on critic 1 and between 100k and 2.35B on critic 2.  This coincided with a rise in the entropy temperature to 40 from zero and brought back down when the entropy temp came down to around 2.k. So the critics are just giving poor value approximations to the policy which thinks it's doing amazing - and the critics are somehow being negatively affected by the entropy tuning despite entropy tuning leading to better results in the gym-match-continuous envs. It may just be that there is nothing to learn, like feeding noise for targets and so the thing is flailing around looking for something that works. So perhaps there's missing info in the observation.

* dd2d.b Giving a speed only reward when above 2m/s and within lane deviation of 1, results in well behaved critic and policy losses. We diverge towards the end of training, but mostly due to reduced variance and get 562/~900 total possible reward. However, not much distance is accrued as the agent learns that turning fast gives it more immediate speed reward and therefore drives off the track every episode.

* dd2d.c Making the reward dependent on distance advancing along route and equal to speed vs gating it with lane_deviation and speed is working *much* better (note we are still on a static map). Not quite sure why this is, perhaps because I didn’t require a minimum speed like before. However, this diverged when the agent started to prefer high speed rewards again over staying on the track for longer.

* dd2d.d In the next experiment, I tried the same distance advance gating, but with zero or one reward. This learned more slowly, but did not diverge. It seems like we should try giving continuous reward that is confined by giving a max of 1 at the max desired speed. I tried this before and it did terribly, but it was gated by lane deviation and minimum speed. However, giving rewards of just 0 or 1, or -1 seems to be a thing in RL.

* So funny thing, the distance based reward didn't take into account whether waypoints on the route had been visited before. So one of the agents learned to go in circles (reverse is disabled) and hit the same way point over and over, getting thousands of points when the max should have been 175! Another run with the same reward however has done really well, getting half way up the track (75 meters).

* dd2d.e So now I'm keeping track of the max distance achieved to prevent the circle cheat, and early training was good reaching 3m by 100k steps, better than 2.3 achieved in the best bot so far, but we then diverge and start to drive extremely slow. With a gamma of 0.99 - we have roughly 100 steps of future reward that we care about. At 60FPS realtime (wall time is faster), we only care about 2-3 seconds ahead. This seems fine as we receive one point roughly every half meter, so maximizing 100 step return should lead to faster progress along the route.

* Aside: training steps per second is ~20 or 1.629e6 steps in 22 hours.
* Env steps per second is 2557: 
    ```
    import deepdrive_zero
    env = gym.make('deepdrive-2d-v0')
    env.reset()
    %time env.step([0.01, 1, 0])
    CPU times: user 391 µs, sys: 37 µs, total: 428 µs
    Wall time: 445 µs
    ```

* 10 actions per second - with 60 physics steps per second allows for 2.5x-5xs faster learning. 75 meters in 1.5M steps for 60aps while in 600k steps for 10aps.

* Large batches just learning slowly, so we could do larger learning rate.

* NaN's in the losses and accel after achieving most of route - not sure why - could be that our finite horizon return of 100-500 steps decreases as the episode will be over. Perhaps then, we should give 1 large reward at the end of the route. Right now, each waypoint is 0.5 meters and gives 1 reward. So we are incentivized at 10FPS to go 0.5 meters every 100ms or 5m/s or 10mph. We end up going twice that pretty consistently... so maybe I'm off by 2 somewhere. 

* Batch size of 1024, lr 0.001 converges 1.5x slower (8.3k[5:37h] test steps vs 5.6k[2:45h]) than default batch 256, lr 0.0003 - single run. Larger batches, 65k/0.002 and 32k/0.001 did not converge in 1 - 2 days of training.

* So we've added minimizing gforce to the objective of achieving waypoint rewards which has been troublesome in the past resulting in the car not going anywhere as the initial acceleration is disincented. So this time, I used a curriculum of speed first, gforce later, which seemed to work. The way I added it was to randomly give gforce penalties proportional to the avg trip pct completed by the agent (I used all time avg, so using a rolling avg would be a good change in the future). So if the agent was completing 1% of the trip on average, and performed a high g turn or deceleration, there'd be a 1% chance of losing either half or all of the frame's distance reward for 0.1g and 0.4g respectively - and for 1g, there'd be a 1% chance of a -1 reward and game over. In this way, I kept a discrete reward which is the only type that worked for distance, while combining two objectives.

* In another experiment, I started including gforce rewards at 50% avg trip completion, but 50% seemed to be too late and showed little signs of minimizing gforce before the policy and critics exploded at 800k episodes. In the experiment above, 2M episodes were completed without exploding gradients.

* Also giving gforce penalties from the start of training results in achieving zero distance after several hours of training.

* After lengthening training time with curriculum gforce (and avoiding exploding gradients with gradient clipping), I was able to get one agent out of three to decrease gforce levels while marginally increasing or maintaining distance. The other 2 runs diverged, dropping to a speed of zero, with one of the runs experiencing exploding loss even with gradient clipping and the other experiencing exponential growth in the critics' and policy loss before dropping to zero speed.

* We see much more divergence (where we regress from a policy that matches the target angle within 5 degrees to one that wobbles around the target angle by ~90deg) in the dd2d env setup, where rewards are just based on matching the angle to goal, than stable-tracking env. Divergence happens with and without automatic entropy tuning.

* Autotuning in angle match dd2d does much better, and reaches perfect angle accuracy several times before 1M steps. Also, the max g-force with autotuning is below jarring 0.4 most of the time and not the case without autotuning which still exhibits harmful g-forces of over 1g ~10% of episodes. Still we diverge and converge several times, hitting harmful g's as we do so.

* Surprisingly, modifying the above experiment to penalize gforce in addition to rewarding angle results in worse angle and gforce performance. Since accel is constant, the network needs to figure out how to provide small corrections to the steering in order to reduce the angle_ahead to the waypoint vs making large corrections and thus large g-force. I only provide angle_ahead as input, but I think that should be enough as reward is based on gforce and angle ahead, and the action (steer) determines those. After looking at the training history, it looks like there *might* be several periods where we converge to an acceptable policy that we could save the network and test with. Next I will do this and also provide prev_steer as input, as changing this too much is what results in large g-forces.

* !!!!!!!!!!!!!!!!!!!!!!!!!! Okay, so using PPO fixed everything for the single waypoint, match angle environment. G-force is perfect, steering is perfect. Also, training at 10FPS and testing at 60FPS works perfectly. AND you can change the waypoint distance from what was trained and go 10x further. This probably works because only the angle to destination and previous steer are input to 32x32 NN. https://github.com/deepdrive/deepdrive-2d/blob/abba502cb29de1b83749fd309424db9b52ad8a15/deepdrive_zero/envs/env.py#L182

* Next step here is to add accel to output. In the previous successful experiment, accel was hardcoded at a low acceptable value for g-force. Now we'll just allow NN to control accel.

* Curriculum training working well in order to train g-force in more complex environments, i.e. with static obstacle avoidance. So train without g-force penalties (except for end of game when over 1.0g), learn to avoid the static obstacle and get to the waypoint, THEN turn on g-force penalties (also I re-initialize Adam optimizer to all zeroes which I'm not sure makes a difference - but essentially it's like a fine-tune rather than a resume train since I wanted to reset momentum and adaptive learning rates).
* Anecdotal observation: A bug which zeroed out the world velocity vector reported to agents, of both itself and other agents, including a relative velocity to other agents, caused <10% trip completion, as compared to 45% with velocity given. Only one seed was tried, but seems significant given the wide disparity and also the fact that the 0-velocity net diverged.
* So it looks like the above is wrong and that actually the issue has to do with constraining the steer / throttle change so that the steering wheel and gas pedal can't teleport arbitrarily far off distances. It seems like we should then try to output control _change_ values instead of absolute control. This way we can de-normalize and the desired action will be a linear function of the actual action, not a clipped one as before. This raises the question though as to why the network can't deal with such a simple constraint. There's definitely a discontinous relationship between input and output and the network is limited to modeling continuous functions. However, it should be able to model a discontinuous function with arbitrary precision. A small reward could be given to promote outputting values that are within the constraint (perhaps if outputting the change doesn't work we should try this).  
* Tried ReLU on intersection with config that otherwise is able to train agents to reach their waypoints ~80% of the time, but with tanh. With Relu, we got a low angle accuracy of around 0.9, but with good speed. Badly performing agents typically get either steering or accel wrong, but rarely both. So a retrain where angle accuracy was good but speed was to low wouldn't surprise me.
* With delta controls, we are maxing out brake and accel and going very slow. Angle accuracy is good, but we collide a lot.
* What does work is letting the agent steer and accelerate through the full range of the vehicle in a single time step, which is basically letting it perform infeasible actions if the time step is fixed. So next, I'm going to try gamma tuning with constrained controls so that the future estimated value does not interfere as much with learning short term consequences of actions which with our mostly per frame reward should be fine. We can always tune it back up. c.f. OpenAI Dota paper. Another thing that's a little more crazy to try is to not fix the time step, but let actions play out, i.e. a high steering angle change just takes more time. This seems like a good idea in theory in that we can explore with larger actions and timesteps, and then refine our actions over time to smaller ones after exploration sufficient to reach the longer term goals like reaching waypoints and not colliding. Perhaps a higher gamma would help also, but at 0.99, ~100 steps, we should be planning 10 seconds into the future so in theory higher gamma should not do anything in the intersection environment which takes 6 to 7 seconds to complete successfully.
* Lowering gamma (0.95, 0.8) and lam (0.835, 0.8) led to quicker learning during the first 50 epochs or so, but eventually converged to worse performance than higher lam - see experiments/intersection_2_agents_lower_gamma
* What does work super well multiplying a `boost_explore` coefficient to action standard deviations when resuming a model! This allowed agents that had a tendency to collide to "unlearn" this by increasing the collision penalty while not needing to retrain from scratch. Without `boost_explore`, increasing the collision penalty or changing any reward for that matter has little to no effect as the action entropy is too low. Tried `boost_explore` of 1.1 which was not enough then 10 which worked well. See deepdrive_zero/experiments/intersection_2_agents_fine_tune_collision_resume_add_comfort5.py

#### Experimental update and summary

* Things we tried that didn't work:

  * Forward rollouts, training on the best or worst. Even for eval this doesn't work well once the policy is biased towards oscillations. Taking the worst rollout ends up with the same performance as not doing rollouts. Taking the best leads to exponentially decreasing returns!

  * Gamma tuning: Lowering gamma at all does not seem to work well, I'm guessing due to gamma being used in Generalized Advantage Estimation as well and introducing too much variance?
 
  * Outputting action deltas and not allowing the next action to be too far from the current (thereby reducing oscillations). In this case exploration is seemingly limited too much to learn to achieve long term goals. However I found a bug with my implementation of this idea that restricted actions 10x more than I wanted, so it may actually still work!
 
* Things that sort of worked:
Shifting advantages to be pessimistic (works well at first, then converges to same performance - probably should be annealed somehow?).

  * Increasing std deviations of actions - this effectively increases exploration and allows a policy to learn new long term goals. 

* Things that have worked:
  * Ending episode on g-force. (Haven't tried jerk yet). Ending the episode instead of giving negative reward allows the policy to still reach high level goals. Recently I haven't been doing this as I was preferring the tunability of negative reward, but I should start adding it again right?
