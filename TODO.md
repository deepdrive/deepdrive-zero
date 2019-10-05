Something is very wrong with the critic losses, varying between 10k and 166M on critic 1 and between 100k and 2.35B on critic 2.  This coincided with a rise in the entropy temperature to 40 from zero and brought back down when the entropy temp came down to around 2.k. So the critics are just giving poor value approximations to the policy which thinks it's doing amazing - and the critics are somehow being negatively affected by the entropy tuning despite entropy tuning leading to better results in the gym-match-continuous envs. It may just be that there is nothing to learn, like feeding noise for targets and so the thing is flailing around looking for something that works. So we should try two things

1) Turn off entropy tuning or make it less aggressive
2) Just give a reward for speed - end of episode is 10k steps. If we can't get this, there's a bug somewhere.e


Add sign to lane deviation to signal right left. Angle ahead prob accounts for this.


Record the distance travelled along route / last map point.

Make map straight.

Make the speed constant.

Allow replaying episode x,y's in arcade
