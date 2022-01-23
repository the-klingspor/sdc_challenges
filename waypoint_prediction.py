import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


def normalize(v):
    norm = np.linalg.norm(v,axis=0) + 0.00001
    return v / norm.reshape(1, v.shape[1])

def curvature(waypoints):
    '''
    ##### TODO #####
    Curvature as  the sum of the normalized dot product between the way elements
    Implement second term of the smoothin objective.

    args: 
        waypoints [2, num_waypoints] !!!!!
    '''
    return np.sum(np.sum((waypoints[:,2:]-waypoints[:,1:-1])*(waypoints[:,1:-1]-waypoints[:,:-2]),axis=0)/np.sqrt(np.sum((waypoints[:,2:]-waypoints[:,1:-1])**2,axis=0) * np.sum((waypoints[:,1:-1]-waypoints[:,:-2])**2,axis=0)))

def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    '''
    Objective for path smoothing

    args:
        waypoints [2 * num_waypoints] !!!!!
        waypoints_center [2 * num_waypoints] !!!!!
        weight_curvature (default=40)
    '''
    # mean least square error between waypoint and way point center
    ls_tocenter = np.mean((waypoints_center - waypoints)**2)

    # derive curvature
    curv = curvature(waypoints.reshape(2,-1))

    return -1 * weight_curvature * curv + ls_tocenter


def waypoint_prediction(roadside1_spline, roadside2_spline, num_waypoints=6, way_type = "smooth"):
    '''
    ##### TODO #####
    Predict waypoint via two different methods:
    - center
    - smooth 

    args:
        roadside1_spline
        roadside2_spline
        num_waypoints (default=6)
        parameter_bound_waypoints (default=1)
        waytype (default="smoothed")
    '''
    if way_type == "center":
        ##### TODO #####
     
        # create spline arguments
        t = np.linspace(0, 1, num_waypoints)
        
        # derive roadside points from spline
        lane_boundary1_points_points = np.array(splev(t,roadside1_spline))
        lane_boundary2_points_points = np.array(splev(t,roadside2_spline))

        # derive center between corresponding roadside points
        center = np.around((lane_boundary1_points_points + lane_boundary2_points_points) / 2)
        way_point_center = np.array([center[0],center[1]])

        # output way_points with shape(2 x Num_waypoints) : (rowsxcolumns)
        return way_point_center
    
    elif way_type == "smooth":
        ##### TODO #####

         # create spline arguments
        t = np.linspace(0, 1, num_waypoints)
        
        # derive roadside points from spline
        lane_boundary1_points_points = np.array(splev(t,roadside1_spline))
        lane_boundary2_points_points = np.array(splev(t,roadside2_spline))

        # derive center between corresponding roadside points
        center = np.around((lane_boundary1_points_points + lane_boundary2_points_points) / 2)
        way_points_center = np.array([center[0],center[1]])
        
        # optimization
        way_points = minimize(smoothing_objective, way_points_center, args=way_points_center.reshape(2*way_points_center.shape[1]))["x"]

        # way_points = minimize(smoothing_objective, 
        #               (way_points_center), 
        #               args=way_points_center)["x"]

        return way_points.reshape(2,-1)


def target_speed_prediction(waypoints, num_waypoints_used=5,
                            max_speed=60, exp_constant=4.5, offset_speed=30):
    '''
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2,num_waypoints]
        num_waypoints_used (default=5)
        max_speed (default=60)
        exp_constant (default=4.5)
        offset_speed (default=30)
    
    output:
        target_speed (float)
    '''
    print(curvature(waypoints))
    curve_factor = -exp_constant * np.abs(num_waypoints_used - 2 - curvature(waypoints))
    target_speed = (max_speed - offset_speed) * np.exp(curve_factor) + offset_speed

    return target_speed