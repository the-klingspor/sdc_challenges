import numpy as np
import torch
import torch.optim as optim
from action import ActionSet, select_exploratory_action, select_greedy_action
from learning import perform_qlearning_step, update_target_net
from model import DQN
from replay_buffer import ReplayBuffer
from schedule import LinearSchedule
from utils import get_state, visualize_training, ModelEMA, check_early_stop
import os
import matplotlib
import time


def evaluate(env, new_actions = None, load_path='agent.t7'):
    """ Evaluate a trained model and compute your leaderboard scores

	NO CHANGES SHOULD BE MADE TO THIS FUNCTION

    Parameters
    -------
    env: gym.Env
        environment to evaluate on
    load_path: str
        path to load the model (.t7) from
    """
    episode_rewards = []
    action_manager = ActionSet()

    if new_actions is not None:
        action_manager.set_actions ( new_actions )

    actions = action_manager.get_action_set()
    action_size = len(actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # These are not the final evaluation seeds, do not overfit on these tracks!
    #seeds = [22597174, 68545857, 75568192, 91140053, 86018367,
    #         49636746, 66759182, 91294619, 84274995, 31531469]
    seeds = [22395174, 68375857, 75568562, 91140123, 56018367,
             49642346, 62345182, 23442619, 84242344, 53453149,
             56795174, 34575857, 75345862, 95340123, 34518367,
             92634566, 34534582, 23445679, 84246574, 34534549]

    # Build & load network
    policy_net = DQN(action_size, device).to(device)
    checkpoint = torch.load(load_path, map_location=device)
    policy_net.load_state_dict(checkpoint)
    policy_net.eval()

    # Iterate over a number of evaluation episodes
    for i in range(len(seeds)):
        env.seed(seeds[i])
        obs, done = env.reset(), False
        obs = get_state(obs)
        if i == 0:
            matplotlib.image.imsave('obs.png', obs[0] / 255 )
        t = 0

        # Run each episode until episode has terminated or 600 time steps have been reached
        episode_rewards.append(0.0)
        while not done and t < 600:
            env.render()
            action_id = select_greedy_action(obs, policy_net, action_size)
            action = actions[action_id]
            obs, rew, done, _ = env.step(action)
            obs = get_state(obs)
            episode_rewards[-1] += rew
            t += 1
        print('episode %d \t reward %f' % (i, episode_rewards[-1]))

    print('---------------------------')
    print(' mean score: %f' % np.mean(np.array(episode_rewards)))
    print(' median score: %f' % np.median(np.array(episode_rewards)))
    print(' std score: %f' % np.std(np.array(episode_rewards)))
    print('---------------------------')


def learn(env,
          lr=1e-4,
          total_timesteps=100000,
          buffer_size=100000,
          exploration_fraction=1.0,
          exploration_final_eps=0.05,
          train_freq=3,
          action_repeat=3,
          batch_size=64,
          learning_starts=1000,
          gamma=0.99,
          target_network_update_freq=1000,
          weight_decay=1e-6,
          new_actions=None,
          model_identifier='agent',
          outdir="",
          use_doubleqlearning=False,
          use_ema=False,
          gas_schedule=False):
    """ Train a deep q-learning model.
    Parameters
    -------
    env: gym.Env
        environment to train on
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to take
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    action_repeat: int
        selection action on every n-th frame and repeat action for intermediate frames
    batch_size: int
        size of a batched sampled from replay buffer for training
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    weight_decay: float
        Weight decay of the Adam optimizer
    model_identifier: string
        identifier of the agent
    use_ema: bool
        Whether to use an exponential moving average
    gas_schedule: bool
        Whether to use a linear schedule for prioritizing gas actions
    """

    # set float as default
    torch.set_default_dtype(torch.float32)
    
    if torch.cuda.is_available():
        print("\nUsing CUDA.")
        print(torch.version.cuda,"\n")
    else:
        print("\nNot using CUDA.\n")

    episode_rewards = [0.0]
    training_losses = []
    action_manager = ActionSet()

    if new_actions is not None:
        print("Set new actions")
        action_manager.set_actions(new_actions)

    actions = action_manager.get_action_set()

    action_size = len(actions)
    print(action_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build networks
    policy_net = DQN(action_size, device).to(device)
    if use_ema:
        target_net = ModelEMA(policy_net)
        target_net.updates = total_timesteps // train_freq
    else:
        target_net = DQN(action_size, device).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

    # Create replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Create optimizer with lr schedule
    optimizer = optim.Adam(policy_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=lr,
                                                    total_steps=total_timesteps // train_freq)

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize environment and get first state
    obs = get_state(env.reset())

    start = time.time()
    frames_in_episode = 0
    n_neg_rewards = 0

    # Iterate over the total number of time steps
    for t in range(total_timesteps):

        # Select action
        action_id = select_exploratory_action(obs, policy_net, actions,
                                              exploration, t, gas_schedule)
        env_action = actions[action_id]

        # policy_net.extract_sensor_values(torch.from_numpy(obs), 1)

        # Perform action fram_skip-times
        for f in range(action_repeat):
            frames_in_episode += 1
            new_obs, rew, done, _ = env.step(env_action)
            early_done, punishment, n_neg_rewards = check_early_stop(rew, n_neg_rewards, episode_rewards[-1],
                                                                     frames_in_episode, exploration=exploration, t=t)
            score = rew
            rew += punishment
            # episode linear reward increment
            if rew > 0:
                rew += 0.2 * frames_in_episode
            episode_rewards[-1] += score
            # episode ended or early stopping
            done = done or early_done
            if done:
                break

            # you can comment out this.
            # env.render()

        # Store transition in the replay buffer.
        new_obs = get_state(new_obs)
        replay_buffer.add(obs, action_id, rew, new_obs, float(done))
        obs = new_obs

        if done:
            # Start new episode after previous episode has terminated
            print("timestep: " + str(t) + " \t reward: " + str(episode_rewards[-1]))
            obs = get_state(env.reset())
            episode_rewards.append(0.0)
            frames_in_episode = 0

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            loss = perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device, use_doubleqlearning, use_ema)
            training_losses.append(loss.detach().cpu())

            if use_ema:
                target_net.update(policy_net)
            # update lr
            scheduler.step()

        if t > learning_starts and not use_ema and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target_net(policy_net, target_net)

        if t % 1000 == 0:
            end = time.time()
            print(f"\n** {t} th timestep - {end - start:.5f} sec passed**\n")

    end = time.time()
    print(f"\n** Total {end - start:.5f} sec passed**\n")

    # Save the trained policy network
    torch.save(policy_net.state_dict(), os.path.join ( outdir, model_identifier+'.t7' ))

    # Visualize the training loss and cumulative reward curves
    visualize_training(episode_rewards, training_losses, model_identifier, outdir)
