import os
import sys
import gym
import argparse
import pyglet
import matplotlib.pyplot as plt
import numpy as np

from pyglet import gl
from pyvirtualdisplay import Display

from gym.envs.box2d.car_racing import CarRacing
from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, TargetSpeedPrediction
from lateral_control import LateralController
from longitudinal_control import LongitudinalController


def evaluate(env):
    """
    """

    # action variables
    a = np.array( [0.0, 0.0, 0.0] )

    # init environement
    env.render()
    env.reset()


    for episode in range(3):
        observation = env.reset()
        # init modules of the pipeline
        LD_module = LaneDetection()
        LatC_module = LateralController(gain_constant=3.0, damping_constant=0.1)
        LongC_module = LongitudinalController(KP=0.5, KI=1e-5, KD=0.2)
        num_waypoints = 6

        TargetSpeed_module = TargetSpeedPrediction(num_waypoints_used=num_waypoints,
                                                   curve_damping_entry=0.05,
                                                   curve_damping_exit=0.02,
                                                   max_speed=80,
                                                   exp_constant=20,
                                                   offset_speed=30)

        reward_per_episode = 0
        for t in range(600):
            # perform step
            s, r, done, speed, info = env.step(a)

            # lane detection
            lane1, lane2 = LD_module.lane_detection(s)

            # waypoint and target_speed prediction
            waypoints = waypoint_prediction(lane1, lane2)
            target_speed = TargetSpeed_module.predict(waypoints)

            # control
            a[0] = LatC_module.stanley(waypoints, speed)
            a[1], a[2] = LongC_module.control(speed, target_speed)

            # reward
            reward_per_episode += r
            env.render()

        print('episode %d \t reward %f' % (episode, reward_per_episode))


def calculate_score_for_leaderboard(env):
    """
    Evaluate the performance of the network. This is the function to be used for
    the final ranking on the course-wide leader-board, only with a different set
    of seeds. Better not change it.
    """
    # action variables
    a = np.array( [0.0, 0.0, 0.0] )

    # init environement
    env.render()
    env.reset()

    seeds = [22597174, 68545857, 75568192, 91140053, 86018367,
             49636746, 66759182, 91294619, 84274995, 31531469]
    
    total_reward = 0

    for episode in range(len(seeds)):
        env.seed(seeds[episode])
        observation = env.reset()

        # init modules of the pipeline
        LD_module = LaneDetection()
        LatC_module = LateralController(gain_constant=3.0, damping_constant=0.1)
        LongC_module = LongitudinalController(KP=0.5, KI=1e-5, KD=0.2)

        num_waypoints = 6

        TargetSpeed_module = TargetSpeedPrediction(num_waypoints_used=num_waypoints,
                                                   curve_damping_entry=0.05,
                                                   curve_damping_exit=0.02,
                                                   max_speed=80,
                                                   exp_constant=20,
                                                   offset_speed=30)


        reward_per_episode = 0
        for t in range(600):
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
            reward_per_episode += r

            env.render()

        print('episode %d \t reward %f' % (episode, reward_per_episode))
        total_reward += np.clip(reward_per_episode, 0, np.infty)

    print('---------------------------')
    print(' total score: %f' % (total_reward / 10))
    print('---------------------------')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument( '--score', action="store_true", help='a flag to evaluate the pipeline for the leaderboard' )
    parser.add_argument( '--display', action="store_true", help='a flag indicating whether training runs in the cluster' )

    args = parser.parse_args()

    if args.display:
        display = Display( visible = 0, size = ( 800, 600 ) )
        display.start()
        print('display started')

    env = CarRacing()

    if args.score:
        calculate_score_for_leaderboard(env)
    else:
        evaluate(env)

    env.close()
    if args.display:
        display.stop()

