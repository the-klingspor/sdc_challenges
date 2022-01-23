import gym
from gym.envs.box2d.car_racing import CarRacing
from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, TargetSpeedPrediction
from lateral_control import LateralController
from longitudinal_control import LongitudinalController
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key

# action variables
a = np.array( [0.0, 0.0, 0.0] )

# init environement
env = CarRacing()
env.render()
env.reset()

# define variables
total_reward = 0.0
steps = 0
restart = False

# init modules of the pipeline
LD_module = LaneDetection()
LatC_module = LateralController(gain_constant=3.0, damping_constant=0.1)
KP = 0.5
KI = 1e-5
KD = 0.2
LongC_module = LongitudinalController(KP, KI, KD)

num_waypoints = 6

TargetSpeed_module = TargetSpeedPrediction(num_waypoints_used=num_waypoints,
                                           curve_damping_entry=0.05,
                                           curve_damping_exit=0.02,
                                           max_speed=80,
                                           exp_constant=20,
                                           offset_speed=30)


# init extra plot
fig = plt.figure()
plt.ion()
plt.show()


while True:
    # perform step
    s, r, done, speed, info = env.step(a)

    # lane detection
    lane1, lane2 = LD_module.lane_detection(s)

    # waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2, num_waypoints=num_waypoints)
    target_speed = TargetSpeed_module.predict(waypoints)

    # control
    a[0] = LatC_module.stanley(waypoints, speed)
    a[1], a[2] = LongC_module.control(speed, target_speed)

    # reward
    total_reward += r

    # outputs during training
    if steps % 2 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("speed {:+0.2f} targetspeed {:+0.2f}".format(speed, target_speed))

        #LD_module.plot_state_lane(s, steps, fig, waypoints=waypoints)
        LongC_module.plot_speed(speed, target_speed, steps, fig)

    steps += 1
    env.render()

    # check if stop
    if done or restart or steps>=600: 
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        break

env.close()