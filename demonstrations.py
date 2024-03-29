import os
import numpy as np
import gym
try:
    from pyglet.window import key
except:
    print("Please note that you can't collect data on the cluster.")


def load_demonstrations(data_folder):
    """
    1.1 a)
    Given the folder containing the expert demonstrations, the data gets loaded and
    stored it in two lists: observations and actions.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    return:
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    # load paths
    files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    obs_paths = []
    action_paths = []

    for f in files:
        if f.startswith("observation"):
            obs_paths.append(f)
        elif f.startswith("action"):
            action_paths.append(f)

    # sort to match observations and actions
    obs_paths.sort()
    action_paths.sort()

    # append directory to file names
    obs_paths = [os.path.join(data_folder, f) for f in obs_paths]
    action_paths = [os.path.join(data_folder, f) for f in action_paths]

    # load numpy files
    observations = [np.load(f) for f in obs_paths]
    actions = [np.load(f) for f in action_paths]

    return observations, actions


def save_demonstrations(data_folder, actions, observations):
    """
    1.1 f)
    Save the lists actions and observations in numpy .npy files that can be read
    by the function load_demonstrations.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """

    # create new directory for this run
    os.makedirs(data_folder, exist_ok=True)
    count = 1
    path = os.path.join(data_folder,str(count))
    creating_dir = True
    while(creating_dir):
        if(os.path.isdir(path)):
            count += 1
            path = os.path.join(data_folder,str(count))
        else:
            os.mkdir(path)
            creating_dir = False

    for index, observation in enumerate(observations):
        np.save(path + "/observation_%05d.npy" % index, observation)
        np.save(path + "/action_%05d.npy" % index, actions[index])
    print("run was saved in " + path)


class ControlStatus:
    """
    Class to keep track of key presses while recording demonstrations.
    """

    def __init__(self):
        self.stop = False
        self.save = False
        self.quit = False
        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

    def key_press(self, k, mod):
        if k == key.ESCAPE: self.quit = True
        if k == key.SPACE: self.stop = True
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


def record_demonstrations(demonstrations_folder):
    """
    Function to record own demonstrations by driving the car in the gym car-racing
    environment.
    demonstrations_folder:  python string, the path to where the recorded demonstrations
                        are to be saved

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    TAB:                save the current run
    """
    env = gym.make('CarRacing-v0').env
    status = ControlStatus()
    total_reward = 0.0

    while not status.quit:
        observations = []
        actions = []
        # get an observation from the environment
        observation = env.reset()
        env.render()

        # set the functions to be called on key press and key release
        env.viewer.window.on_key_press = status.key_press
        env.viewer.window.on_key_release = status.key_release

        while not status.stop and not status.save and not status.quit:
            # collect all observations and actions
            observations.append(observation.copy())
            actions.append(np.array([status.steer, status.accelerate,
                                     status.brake]))
            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, info = env.step([status.steer,
                                                        status.accelerate,
                                                        status.brake])
            total_reward += reward
            env.render()

        if status.save:
            save_demonstrations(demonstrations_folder, actions, observations)
            status.save = False

        status.stop = False
        env.close()


#load_demonstrations("/home/joschi/Documents/Studium/WS2122/Self-driving Cars/sdc_challenges/data/teacher_new/")