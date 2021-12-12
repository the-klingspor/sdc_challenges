import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import math

from copy import deepcopy

mpl.use('Agg')


def get_state(state): 
    """ Helper function to transform state """ 
    state = np.ascontiguousarray(state, dtype=np.float32) 
    return np.expand_dims(state, axis=0)


def visualize_training(episode_rewards, training_losses, model_identifier, ourdir=""):
    """ Visualize training by creating reward + loss plots
    Parameters
    -------
    episode_rewards: list
        list of cumulative rewards per training episode
    training_losses: list
        list of training losses
    model_identifier: string
        identifier of the agent
    """
    n_avg = 100
    episode_rewards = np.array(episode_rewards)
    plt.plot(episode_rewards)
    plt.plot(moving_average(episode_rewards, n_avg), color="r")
    plt.savefig(os.path.join(ourdir, "episode_rewards-"+model_identifier+".png"))
    plt.close()

    training_losses = np.array(training_losses)
    plt.plot(training_losses)
    plt.plot(moving_average(training_losses, n_avg), color="r")
    plt.savefig(os.path.join(ourdir, "training_losses-"+model_identifier+".png"))
    plt.close()


def moving_average(a, n=3):
    a_padded = np.pad(a, (n//2, n-1-n//2), mode='median')
    a_smooth = np.convolve(a_padded, np.ones(n)/n, mode='valid')
    return a_smooth


def check_early_stop(rew, n_neg_rewards, frames_in_episode, total_rewards,
                     max_neg_rewards=100):
    """ Check if the episode should be stopped early because of poor performance.

    Parameters:
    -------
    rew: float
        This frames's reward.
    n_neg_rewards: int
        Number of consecutive frames with negative reward.
    frames_in_episode: int
        Number of frames of this episode seen so far
    total_rewards: float
        The total amount of rewards for this episode so far
    max_neg_rewards: int

    Returns
    -------
    early_done: bool
        Whether to stop the episode early because of poor results
    punishment: float
        A punishment term for the reward, if early stopping was necessary
    n_neg_rewards: int
        Number of consecutive frames with negative reward.
    """
    early_done = False
    punishment = 0
    # increase counter and check for early stop, if negative reward
    if frames_in_episode > 10 and rew < 0:
        n_neg_rewards += 1

        if n_neg_rewards > max_neg_rewards:
            early_done = True
            if total_rewards <= 500:
                punishment = -20.0
            n_neg_rewards = 0

    # reset counter if positive reward
    else:
        n_neg_rewards = 0

    return early_done, punishment, n_neg_rewards


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


def is_parallel(model):
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)
