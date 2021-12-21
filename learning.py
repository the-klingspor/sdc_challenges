import numpy as np
import torch
import torch.nn.functional as F
import gc


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device, use_doubleqlearning=False, use_ema=False, PRB=False):
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
    use_ema: bool
        whether an exponential moving average is used as target network

    Returns
    -------
    float
        loss value for current learning step
    """
 
    # 1. Sample transitions from replay_buffer
    [obs_batch, act_batch, rew_batch, next_obs_batch, done_mask] = replay_buffer.sample(batch_size)
    obs_batch = torch.tensor(obs_batch)
    rew_batch = torch.tensor(rew_batch)
    next_obs_batch = torch.tensor(next_obs_batch)
    done_mask = torch.tensor(done_mask)
    act_batch = torch.tensor(act_batch, dtype=torch.long)[:, None]
    if torch.cuda.is_available():
            obs_batch = obs_batch.cuda()
            rew_batch = rew_batch.cuda()
            next_obs_batch = next_obs_batch.cuda()
            done_mask = done_mask.cuda()
            act_batch = act_batch.cuda()

    # 2. Compute Q(s_t, a)
    prediction = policy_net(obs_batch).gather(1, act_batch)  # [np.arange(batch_size), act_batch]

    # 3. Compute \max_a Q(s_{t+1}, a) for all next states.
    with torch.no_grad():
        if use_ema:
            q_values = target_net.ema(next_obs_batch)
        else:
            q_values = target_net(next_obs_batch)
        if use_doubleqlearning:
            next_q_values = policy_net(next_obs_batch)
            next_state_q_values = q_values
            # choose target action based on maximum next state policy action
            policy_actions = torch.max(next_q_values, 1)[1][:, None]
            expected_q = next_state_q_values.gather(1, policy_actions).squeeze()
        else:
            expected_q = torch.amax(q_values, 1)

    # 4. Mask next state values where episodes have terminated
    expected_q *= 1 - done_mask

    # 5. Compute the target
    target = rew_batch + gamma * expected_q
    target = target[:, None]

    # 6. Compute the loss
    #loss = ((prediction - target)**2).mean()
    loss = F.smooth_l1_loss(prediction, target)
    # MSE loss
    #loss_f = torch.nn.MSELoss()
    #loss = loss_f(prediction, target)

    # 7. Calculate the gradients
    optimizer.zero_grad()
    loss.backward()
            
    # 8. Clip the gradients
    #torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    #or
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    # 9. Optimize the model
    optimizer.step()

    # ------------------- update priorities ------------------- #
    if(PRB){
        delta = abs(target - prediction).numpy()
        #print("delta", delta)      
        self.memory.update_priorities(delta, indices)  
    }

    return loss


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
