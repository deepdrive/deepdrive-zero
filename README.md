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
