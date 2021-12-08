import random
import torch
import numpy as np


def select_greedy_action(state, policy_net, action_set):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_set: ActionSet
        Set class containing valid actions
    Returns
    -------
    int
        ID of selected action
    """
    state = torch.tensor(state)
    if torch.cuda.is_available():
        state = state.cuda()
    with torch.no_grad():
        x = policy_net(state)
    return int(torch.argmax(x))


def select_exploratory_action(state, policy_net, action_set, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_set: ActionSet
        Set class containing valid actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """
    action_size = len(action_set.actions)
    if exploration.value(t) >= random.uniform(0, 1):
        # over selection of gas actions
        gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in action_set.actions])
        action_weights = 14.0 * gas_actions + 1.0
        action_weights /= np.sum(action_weights)
        return np.random.choice(action_size, p=action_weights)
    else:
        return select_greedy_action(state, policy_net, action_set)


class ActionSet:
    
    def __init__(self):
        """ Initialize actions
        """
        self.actions = [[-1.0, 0.05, 0], [1.0, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0]]

    def set_actions(self, new_actions):
        """ Set the list of available actions
        Parameters
        ------
        new_actions: list
            list of available actions
        """
        self.actions = new_actions

    def get_action_set(self):
        """ Get the list of available actions
        Returns
        -------
        list
            list of available actions
        """
        return self.actions
