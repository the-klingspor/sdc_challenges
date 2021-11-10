import torch
import torch.nn as nn
import random
import time
import math

from network import ClassificationNetwork
from demonstrations import load_demonstrations


def train(data_folder, trained_network_file):
    """
    Function for training the network.
    """
    infer_action = ClassificationNetwork()
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=3e-4)
    #loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()

    observations, actions = load_demonstrations(data_folder)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    #batches = [batch for batch in zip(observations,
    #                                  infer_action.actions_to_classes(actions))]
    # use regression training instead of binary classes
    batches = [batch for batch in zip(observations,
                                      infer_action.actions_to_regs(actions))]

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nr_epochs = 100
    batch_size = 64
    start_time = time.time()

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: (((1 + math.cos(x * math.pi / nr_epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(device))
            batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0),
                                         (-1, 2))

                batch_out = infer_action(batch_in)
                loss = loss_function(batch_out, batch_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss

                batch_in = []
                batch_gt = []

        scheduler.step()

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))

    torch.save(infer_action, trained_network_file)
