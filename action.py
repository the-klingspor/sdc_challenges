import random
import torch


def select_greedy_action(state, policy_net, action_size):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
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

def select_exploratory_action(state, policy_net, action_size, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """

    if(exploration.value(t) >= random.uniform(0, 1)):
        return random.randrange(action_size)
    else:
        state = torch.tensor(state)
        if torch.cuda.is_available():
                state = state.cuda()
        with torch.no_grad():        
            x = policy_net(state)
        return int(torch.argmax(x))

class ActionSet:
    
    def __init__( self ):
        """ Initialize actions
        """
        self.actions = [[-1.0, 0.05, 0], [1.0, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0]]

    def set_actions (self, new_actions):
        """ Set the list of available actions
        Parameters
        ------
        list
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
