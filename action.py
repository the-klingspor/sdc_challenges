import random
import torch
import numpy as np


def select_greedy_action(state, policy_net, actions):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    actions: list
        list of all valid actions
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
        x = x.max(1)[1].squeeze().cpu()
    return x


def select_exploratory_action(state, policy_net, actions, exploration, t, gas_schedule=False):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    actions: list
        list of all valid actions
    exploration: linearschedule
        linear exploration schedule
    t: int
        current time-step
    returns
    -------
    int
        id of selected action
    """
    action_size = len(actions)
    if exploration.value(t) >= random.uniform(0, 1):
        # over selection of gas actions
        gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in actions])
        gas_weight = exploration.value(t) if gas_schedule else 1.0
        action_weights = 14 * gas_weight * gas_actions + 1.0
        action_weights /= np.sum(action_weights)
        return np.random.choice(action_size, p=action_weights)
    else:
        return select_greedy_action(state, policy_net, actions)


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
