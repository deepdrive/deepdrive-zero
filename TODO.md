Write gforce to tfx, to get an idea of average instead of just max.

Do gradient clipping using std deviation of recent 100 gradient norms instead of hardcoding at 100.

Max gforce of 200 seems wrong, maybe
during a GC or something IDK - or perhaps it's a problem with reset.

Gforce
- Add g-force to reward, -1 if we over 0.4
- Record gforce levels to tensorboard


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
