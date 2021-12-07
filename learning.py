import numpy as np
import torch
import torch.nn.functional as F


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device, use_doubleqlearning = False):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """
 
    # 1. Sample transitions from replay_buffer
    [obs_batch, act_batch, rew_batch, next_obs_batch, done_mask] = replay_buffer.sample(batch_size)
    obs_batch = torch.tensor(obs_batch)
    if torch.cuda.is_available():
            obs_batch = obs_batch.cuda()

    # 2. Compute Q(s_t, a)
    prediction = policy_net(obs_batch)[np.arange(batch_size),act_batch]
    
    # 3. Compute \max_a Q(s_{t+1}, a) for all next states.
    MaxQ = torch.max(target_net(obs_batch),axis=1)

    # 4. Mask next state values where episodes have terminated


    # 5. Compute the target
    target = rew_batch + gamma*MaxQ

    # 6. Compute the loss
    Loss = (target - prediction)**2

    # 7. Calculate the gradients
    #stochastic gradient descent

    # 8. Clip the gradients

    # 9. Optimize the model

    # Tip: You can use use_doubleqlearning to switch the learning modality.

    return Loss

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network
    target_net.load_state_dict(policy_net.state_dict())
