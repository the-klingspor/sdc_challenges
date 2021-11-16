import os
import numpy as np
import gym
import torch
import torch.nn as nn
import random
import time
import random as rand

import numpy as np
import matplotlib.pyplot as plt

from network import ClassificationNetwork
from demonstrations import save_demonstrations
try:
    from pyglet.window import key
except:
    print("Please note that you can't collect data on the cluster.")


class ControlStatus:
    """
    Class to keep track of key presses while recording demonstrations.
    """

    def __init__(self):
        self.stop = False
        self.reload = False
        self.save = False
        self.quit = False
        self.Continue = False
        self.blocking = False
        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

    def key_press(self, k, mod):
        if k == key.ESCAPE: self.quit = True
        if k == key.SPACE: self.stop = True
        if k == key.R: self.reload = True
        if k == key.C: self.Continue = True
        if k == key.TAB: self.save = True
        if k == key.LEFT: self.steer = -1.0
        if k == key.RIGHT: self.steer = +1.0
        if k == key.UP: self.accelerate = +0.5
        if k == key.DOWN: self.brake = +0.8

    def key_release(self, k, mod):
        if k == key.LEFT and self.steer < 0.0: self.steer = 0.0
        if k == key.RIGHT and self.steer > 0.0: self.steer = 0.0
        if k == key.UP: self.accelerate = 0.0
        if k == key.DOWN: self.brake = 0.0


def record_demonstrations_DAgger(demonstrations_folder,beta,trained_network_file,only_critical):
    """
    Dagger-like-Function to record own demonstrations by driving the car with network support in the gym car-racing
    environment.
    demonstrations_folder:  python string, the path to where the recorded demonstrations
                        are to be saved
    beta: measure of how souverain the network is driving (beta = 1 means no controll for the network, beta = 0 no controll for the expert)
    trained_network_file: location of the network that should be used in this recording

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    TAB:                save the current run
    R:                  Reload the exact same track
    C:                  continue past the 600 frame limit
    """
    print("press TAB to save, SPACE to restart, or ESC to stop")
    print(only_critical)
    infer_action = torch.load(trained_network_file)
    infer_action.eval()
        # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_action.to(device)

    env = gym.make('CarRacing-v0').env
    status = ControlStatus()
    total_reward = 0.0
    seed = rand.randint(00000000,99999999)

    observations = []
    actions = []
    # get an observation from the environment
    observation = env.reset()
    env.seed(seed)
    env.render()

    # set the functions to be called on key press and key release
    env.viewer.window.on_key_press = status.key_press
    env.viewer.window.on_key_release = status.key_release

    while not status.quit:
        
        frames = 0
        score = 0.0
        while not status.stop and not status.save and not status.quit and not status.blocking and not status.reload:
            frames += 1

            action_scores = infer_action(torch.Tensor(
                np.ascontiguousarray(observation[None])).to(device))
            agent_action = infer_action.scores_to_action(action_scores)

            action = (1-beta) * np.array([status.steer,status.accelerate,status.brake]) + beta * np.array(agent_action)

            # collect all observations and actions
            observations.append(observation.copy())
            actions.append(np.array(action))

            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, info = env.step(action)

            score += reward
            env.render()

            if frames > 650:
                print("score: ",score)
                status.blocking = True
        env.render()
        if status.save:
            env.close()
            if only_critical:
                [reduced_observations,reduced_actions] = reduce_to_critical_Curves(observations,actions)
                save_demonstrations(demonstrations_folder, reduced_actions, reduced_observations)
                print("only critical data saved")
            else:
                save_demonstrations(demonstrations_folder, actions, observations)
            observations = []
            actions = []
            # get an observation from the environment
            seed = rand.randint(00000000,99999999)
            env.seed(seed)
            observation = env.reset()
            env.render()
            # set the functions to be called on key press and key release
            env.viewer.window.on_key_press = status.key_press
            env.viewer.window.on_key_release = status.key_release
            status.blocking = False
            status.save = False

        if status.reload:
            env.close()
            save_demonstrations(demonstrations_folder, actions, observations)
            observations = []
            actions = []
            # get an observation from the environment
            env.seed(seed)
            observation = env.reset()
            env.render()
            # set the functions to be called on key press and key release
            env.viewer.window.on_key_press = status.key_press
            env.viewer.window.on_key_release = status.key_release
            status.blocking = False
            status.reload = False
        
        if status.Continue:
            frames = 0
            status.blocking = False
            status.Continue = False

        if status.stop:
            env.close()
            observations = []
            actions = []
            # get an observation from the environment
            seed = rand.randint(00000000,99999999)
            env.seed(seed)
            observation = env.reset()
            env.render()
            # set the functions to be called on key press and key release
            env.viewer.window.on_key_press = status.key_press
            env.viewer.window.on_key_release = status.key_release
            status.blocking = False
            status.stop = False

        if status.quit:
            env.close()

def reduce_to_critical_Curves(observations,actions):
    """
    deletes all oberservations which don't have the red curve markings below the y_top_threshold
    """
    y_top_threshold = 60 #pixel

    observations = np.array(observations)
    delete = []
    for i in range(observations.shape[0]):
        if(np.amax(observations[i,y_top_threshold:83,:,0]) != 255):
            delete.append(i)
    reduced_observations = np.delete(observations,delete,axis=0)
    reduced_actions = np.delete(actions,delete,axis=0)
    return [reduced_observations,reduced_actions]

def inspect_observation_with_values(observations):
    import matplotlib.cm as cm

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(observations)
    print("max red value: ",np.amax(observations[40:83,:,0]))
    numrows = observations.shape[0]
    numcols = observations.shape[1]
    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col>=0 and col<numcols and row>=0 and row<numrows:
            z = observations[row,col]
            return 'x=%1.4f, y=%1.4f, rot=%3d, gruen=%3d, blau=%3d'%(x, y, z[0], z[1], z[2])
        else:
            return 'x=%1.4f, y=%1.4f'%(x, y)
    ax.format_coord = format_coord
    plt.show()


## inspect a picture with the r,g,b values
#inspect_observation_with_values(np.load(r"/home/lenny/Uni/SDC/ex_01_imitation_learning/sdc_challenges/data/teacher/observation_11.npy"))

## showcase the removal of non critical data in the teacher dataset
# from demonstrations import load_demonstrations 
# from demonstrations import save_demonstrations 
# [observations,actions] = load_demonstrations(r"/home/lenny/Uni/SDC/ex_01_imitation_learning/sdc_challenges/data/teacher")
# [reduced_observations,reduced_actions] = reduce_to_critical_Curves(observations,actions)
# save_demonstrations(r"/home/lenny/Uni/SDC/ex_01_imitation_learning/sdc_challenges/data/teacher_critical", reduced_actions, reduced_observations)
