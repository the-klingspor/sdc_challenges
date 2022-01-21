import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LateralController:
    '''
    Lateral control using the Stanley controller

    functions:
        stanley 

    init:
        gain_constant (default=5)
        damping_constant (default=0.5)
    '''


    def __init__(self, gain_constant=5, damping_constant=0.6):

        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0


    def stanley(self, waypoints, speed):
        '''
        one step of the stanley controller with damping
        args:
            waypoints (np.array) [2, num_waypoints]
            speed (float)
        '''
        waypoints = waypoints.T
        waypoint_1 = waypoints[0]
        waypoint_2 = waypoints[1]
        waypoint_orientation = waypoint_2 - waypoint_1
        waypoint_orientation /= np.linalg.norm(waypoint_orientation)
        car_position = np.array([47.5, 0])  # always static at same position
        car_orientation = np.array([0, 1.0])  # always shows directly to the top

        # derive orientation error as the angle of the first path segment to the car orientation
        orientation_error = np.arccos(np.clip(np.dot(waypoint_orientation, car_orientation), -1.0, 1.0))

        # derive cross track error as distance between desired waypoint at spline parameter equal zero to the car position
        cross_track_error = np.cross(waypoint_orientation, car_position - waypoint_1)
        cross_track_error /= np.linalg.norm(waypoint_orientation)

        # derive stanley control law
        # prevent division by zero by adding a small epsilon
        eps = 1e-6
        cross_track_term = self.gain_constant * cross_track_error / (speed + eps)
        steering_angle = orientation_error + np.arctan(cross_track_term)

        # derive damping term
        steering_angle -= self.damping_constant * (steering_angle - self.previous_steering_angle)

        # save steering angle
        self.previous_steering_angle = steering_angle

        # clip to the maximum stering angle (0.4) and rescale the steering action space
        return np.clip(steering_angle, -0.4, 0.4) / 0.4






