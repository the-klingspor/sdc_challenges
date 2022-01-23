import gym
from gym.envs.box2d.car_racing import CarRacing

from lane_detection import LaneDetection
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key

# action variables
a = np.array( [0.0, 0.0, 0.0] )

# define keys
def key_press(k, mod):
    global restart
    if k==0xff0d: restart = True
    if k==key.LEFT:  a[0] = -1.0
    if k==key.RIGHT: a[0] = +1.0
    if k==key.UP:    a[1] = +1.0
    if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    if k==key.R:    
        print('stop')

def key_release(k, mod):
    if k==key.LEFT  and a[0]==-1.0: a[0] = 0
    if k==key.RIGHT and a[0]==+1.0: a[0] = 0
    if k==key.UP:    a[1] = 0
    if k==key.DOWN:  a[2] = 0

# init environement
env = CarRacing()
env.render()
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release
env.reset()

# define variables
total_reward = 0.0
steps = 0
restart = False

# init modules of the pipeline
LD_module = LaneDetection()

# init extra plot
fig = plt.figure()
plt.ion()
plt.show()
count = 0
while True:
    # perform step
    s, r, done, speed, info = env.step(a)
    # if(count % 100 == 0):
    #     plt.imshow(s[:68,:,:])
    #     plt.show()
    #     grey = LD_module.cut_gray(s)
    #     #plt.imshow(grey)
    #     plt.show()
    # count += 1
    # lane detection
    #[splines,gradient_sum,lane_points] = LD_module.lane_detection(s)
    LD_module.lane_detection(s)
    
    # reward
    total_reward += r

    # outputs during training
    if steps % 4 == 0 or done:
        #print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        #print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        #LD_module.plot_state_lane(s, steps, fig, gradient_sum,lane_points)
        LD_module.plot_state_lane(s, steps, fig)
    steps += 1
    env.render()
    
    if done or restart: break

env.close()