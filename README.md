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


Architecture

The Arcade package is used for rendering, but is decoupled from physics so as
to allow running the sim quickly without rendering. We use Numba to speed
up physics calculations which allows iterating in Python (with the option
to do ahead of time compilation on predefined numeric types) while maintaining
the performance of a compiled binary. 
